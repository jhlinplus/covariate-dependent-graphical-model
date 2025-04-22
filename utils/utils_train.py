import importlib
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC

from .early_stopper import EarlyStopping
from .utils_logging import get_logger

logger = get_logger()

def fix_seed(SEED):

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return

def create_one_dataloader(data_str, dataset_classname, batch_size, num_workers=4, mode='test', size=None, shuffle=None, drop_last=None):
    
    available_datasets = importlib.import_module('utils.datasets')
    datasetClass = getattr(available_datasets, dataset_classname)
    dataset = datasetClass(data_str=data_str, file_type=mode, total_size=size)
    
    if shuffle is None:
        shuffle=True if mode == 'train' else False
    
    if drop_last is None:
        drop_last=False if mode == 'test' else True
    
    return DataLoader(dataset,
                batch_size=batch_size,
                num_workers=min(num_workers,os.cpu_count()-1),
                shuffle=shuffle,
                drop_last=drop_last,
                persistent_workers=False)


def prepare_train_val_test_dataloaders(data_configs, train_configs):
    
    dataset_classname = data_configs['dataset_class']
    batch_size = train_configs['batch_size']
    
    dataloaders = []
    for mode in ['train', 'val', 'test']:
        dataloader = create_one_dataloader(data_configs['data_str'],
                                    dataset_classname,
                                    batch_size,
                                    num_workers=4,
                                    mode=mode, 
                                    size=train_configs['train_size'] if mode=='train' else train_configs[f'num_{mode}'])
        print(f'dataloader for mode={mode} created; num_batches={len(dataloader)}, num_samples={len(dataloader.dataset)}')
        dataloaders.append(dataloader)
        
    return dataloaders

def initialize_model_and_optimizer(network_configs, opt_configs, device):
    
    ## initialize model
    available_models = importlib.import_module('models')
    modelClass = getattr(available_models, network_configs['model_name'])
    model = modelClass(network_configs)
    model.to(device)
    
    logger.info(f"model {network_configs['model_name']} initialized on device {model.dummy.device}")
    model.print_summary()

    ## initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_configs['learning_rate'], weight_decay=opt_configs.get('weight_decay',1.0e-6))
    
    ## setup scheduler
    scheduler = None
    if opt_configs['scheduler_type'] == 'MultiStepLR':
        assert ('milestones' in opt_configs) and ('gamma' in opt_configs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=opt_configs['milestones'],
                                    gamma=opt_configs['gamma'])
    elif opt_configs['scheduler_type'] == 'StepLR':
        assert ('step_size' in opt_configs) and ('gamma' in opt_configs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                    step_size=opt_configs['step_size'],
                                    gamma=opt_configs['gamma'])
    else:
        pass
    
    return model, optimizer, scheduler


def train_epoch(model, optimizer, scheduler, criterion, data_loader, meter=None, gradient_clip_val=1):
    
    torch.set_grad_enabled(True)
    model.train()
    
    losses = []
    for batch_data in data_loader:
        
        x = batch_data['x'].to(model.dummy.device)
        z = batch_data['z'].to(model.dummy.device)
        
        pred, beta = model(x, z)
        loss = criterion(pred, x)
        losses.append(loss.item())

        if meter:
            meter.update(beta.abs(), 1*(batch_data['graph']!=0).to(model.dummy.device))

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    metric = meter.compute() if meter else None
    return float(np.mean(losses)), metric


def val_epoch(model, criterion, data_loader, meter=None):
    
    torch.set_grad_enabled(False)
    model.eval()

    losses = []
    for batch_data in data_loader:
        
        x = batch_data['x'].to(model.dummy.device)
        z = None if 'z' not in batch_data else batch_data['z'].to(model.dummy.device)

        pred, beta = model(x, z)
        loss = criterion(pred, x)
        losses.append(loss.item())
        
        if meter:
            meter.update(beta.abs(), 1*(batch_data['graph']!=0).to(model.dummy.device))
    
    metric = meter.compute() if meter else None
    return float(np.mean(losses)), metric


def test_epoch(model, data_loader):
    
    torch.set_grad_enabled(False)
    model.eval()

    predictions, graphs = [], []
    for batch_data in data_loader:
        
        x = batch_data['x'].to(model.dummy.device)
        z = None if 'z' not in batch_data else batch_data['z'].to(model.dummy.device)

        pred, beta = model(x, z)
        graph = model.unflatten_graph(beta)
        
        predictions.append(pred.cpu().detach())
        graphs.append(graph.cpu().detach())
        
    preds = torch.concat(predictions).numpy() # [batch_size, num_nodes]
    graphs = torch.concat(graphs).numpy() # [batch_size, num_nodes, num_nodes]
    return preds, graphs


def model_fit(model, optimizer, scheduler, train_configs, train_dataloader, val_dataloader=None, output_dir=None, use_meter=False):

    ## get criterion for loss calculation
    criterion = nn.MSELoss()
    
    ## set up early stopping
    use_early_stopping = False
    if train_configs.get('es_patience', None):
        use_early_stopping = True
        assert 'es_monitor' in train_configs, f'early stopping is activated but `es_monitor` missing'
        assert train_configs['es_monitor'] in ['train_loss','val_loss']
        if train_configs['es_monitor'] == 'val_loss':
            assert val_dataloader is not None, f'`es_monitor` has been set to val_loss but val_dataloader is None'
        logger.info(f'early stopping is activated with patience={train_configs["es_patience"]}, monitor={train_configs["es_monitor"]}')
        if not output_dir:
            output_dir = os.path.join('var','tmp',os.environ.get('USER','UNK_USER'))
        logger.info(f'ckpt_dir={output_dir}')
        early_stopper = EarlyStopping(patience=train_configs['es_patience'], min_delta=1.0e-4, ckpt_dir=output_dir, verbose=False)
    
    if use_meter:
        train_meter = AUROC(task="binary").to(model.dummy.device)
        val_meter = AUROC(task="binary").to(model.dummy.device)
    else:
        train_meter, val_meter = None, None
    
    ## start training/validation
    t_start = time.monotonic()
    for epoch in range(train_configs['max_epochs']+1):
        t0 = time.monotonic()
        ## train epoch
        train_loss, train_auroc = train_epoch(model, optimizer, scheduler, criterion, train_dataloader,
                                              meter=train_meter, gradient_clip_val=train_configs['gradient_clip_val'])
        if train_meter:
            train_meter.reset()
        
        if use_early_stopping and train_configs['es_monitor'] == 'train_loss':
            early_stopper(train_loss, model.state_dict())
        
        ## validation epoch
        val_str = ''
        if val_dataloader is not None:
            val_loss, val_auroc = val_epoch(model, criterion, val_dataloader, meter=val_meter)
            if val_meter:
                val_meter.reset()
            if use_early_stopping and train_configs['es_monitor'] == 'val_loss':
                early_stopper(val_loss, model.state_dict())
            val_str = f'val_loss={val_loss:.4f}'
            val_meter_str = f', val_auroc={val_auroc:.3f}' if val_meter else ''
        
        ## verbose
        if train_configs['verbose'] > 0 and (epoch == 0 or (epoch+1) % train_configs['verbose'] == 0):
            train_meter_str = f', train_auroc={train_auroc:.3f}' if train_meter else ''
            print(f'>> Epoch={epoch+1:03d}, lr={float(optimizer.param_groups[0]["lr"]):.1E}; train_loss={train_loss:.4f}{train_meter_str}; {val_str}{val_meter_str}; [timer={time.monotonic()-t0:.0f}s/epoch]')
        ## early stop
        if use_early_stopping and early_stopper.early_stop and epoch >= train_configs.get('min_epochs', 0):
            break
            
    t_end = time.monotonic()
    logger.info(f'model training completed; total time elapsed = {(t_end-t_start)/60:.2f} mins')
    
    return


def model_predict(model, test_dataloader):

    preds, graphs = test_epoch(model, test_dataloader)
    return preds, graphs


def model_fit_predict_snapshot_ensemble(model, optimizer, scheduler, opt_configs, train_dataloader, val_dataloader, test_dataloader, verbose=10):
    ## get criterion for loss calculation
    criterion = nn.MSELoss()
    ## start training/validation/ensemble
    t0 = time.monotonic()
    output_snapshots = []
    for epoch in range(opt_configs['max_epochs']+1):
        train_loss = train_epoch(model, optimizer, scheduler, criterion, train_dataloader, opt_configs['gradient_clip_val'])
        ## perform validation
        if val_dataloader is not None:
            val_loss = val_epoch(model, criterion, val_dataloader)
        if verbose > 0 and (epoch == 0 or (epoch+1)%verbose == 0):
            print(f'>> Epoch={epoch:03d}, train_loss={train_loss:.4f}; lr_base={float(optimizer.param_groups[0]["lr"]):.1E}, lr_fc={float(optimizer.param_groups[1]["lr"]):.1E}')
        ## perform test
        if (epoch !=0) and ((epoch+1) % opt_configs['snapshot_ensemble_interval'] == 0):
            preds, graphs = test_epoch(model, test_dataloader)
            print(f'** Epoch={epoch:03d}, captured one test snapshot')
            output_snapshots.append(graphs)
    logger.info(f'Model training completed; total time elapsed = {(time.monotonic()-t0)/60:.2f} mins')
    output_ensembled = np.concatenate(output_snapshots,axis=-1).mean(axis=-1)
    return output_ensembled


def run_model_pipeline(configs, device, run_seed, output_dir, meter_flag=True):
    
    if not output_dir:
        logger.warning('output_dir is None; no output will be saved')
        
    fix_seed(run_seed)

    train_dl, val_dl, test_dl = prepare_train_val_test_dataloaders(configs['data_configs'], configs['train_configs'])
    model, optimizer, scheduler = initialize_model_and_optimizer(configs['network_configs'], configs['opt_configs'], device)
    
    model_fit(model, optimizer, scheduler, configs['train_configs'], train_dl, val_dl, output_dir, use_meter=meter_flag)
    preds, graphs = model_predict(model, test_dl)
    
    if output_dir is not None:
        np.save(os.path.join(output_dir, 'test_x.npy'), preds)
        np.save(os.path.join(output_dir, 'test_graphs.npy'), graphs)
    
    return model, optimizer
