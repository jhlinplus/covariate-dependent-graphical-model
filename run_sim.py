#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('high')
import yaml

from utils import utils_train, utils_logging, utils_eval

logger = utils_logging.get_logger()


def main(args):

    ###########################################
    ## I. Basic Setup
    ###########################################
    logger.info(f'initial setup')
    suffix = '' if (not args.version) or (not len(args.version)) else '_' + args.version
    config_file = args.config
    if not args.config:
        config_file = os.path.join('configs', f'{args.data_str}{suffix}.yaml')
        if not os.path.exists(config_file):
            config_file = os.path.join('configs', 'competitors', f'{args.data_str}{suffix}.yaml')
        assert os.path.exists(config_file), f'config_file={config_file} does not exist'
    logger.info(f'config file in use={config_file}')
    
    ## load configs
    with open(config_file) as fp:
        configs = yaml.safe_load(fp)
    update_configs_from_args_(configs, args)
    
    ## setup output_dir unless override is provided
    output_dir = args.output_dir or get_output_dir_name(configs, suffix)
    if os.path.exists(output_dir):
        logger.warning(f'output_dir={output_dir} already existed; deleting it')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.info(f'output_dir={output_dir} created')
    
    ## save a copy of everything into args.json for reference
    args.config = config_file
    for section_key, val in configs.items():
        setattr(args, section_key, val)
    with open(os.path.join(output_dir, 'args.json'),'w') as handle:
        json.dump(vars(args), handle, indent=4, default=str)
        logger.info(f'args saved to {output_dir}/args.json')

    ###########################################
    ## II. init network, train, inference
    ###########################################
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else 'cpu'
    logger.info(f'device in use={device}')
    model, optimizer = utils_train.run_model_pipeline(configs, device, args.seed, output_dir)
    if args.debug:
        run_debug_on_train(model, configs, output_dir)
    
    ###########################################
    ## III. simple eval
    ###########################################
    run_simple_eval(output_dir, configs['data_configs']['data_str'], data_key='test', graph_key='graph')
    run_simple_eval(output_dir, configs['data_configs']['data_str'], data_key='test', graph_key='symSkel')
        
    if args.debug:
        run_simple_eval(output_dir, configs['data_configs']['data_str'], data_key='train', graph_key='graph')
        run_simple_eval(output_dir, configs['data_configs']['data_str'], data_key='train', graph_key='symSkel')
    
    return 0


def update_configs_from_args_(configs, args):

    configs['data_configs']['data_str'] = f'{args.data_str}_seed{args.data_seed}'
    configs['train_configs']['train_size'] = args.train_size
    
    del args.data_str, args.data_seed, args.train_size
    return


def get_output_dir_name(configs, suffix):

    model_str = configs['network_configs']['model_name']
    data_str = configs['data_configs']['data_str']
    return os.path.join('output_sim',f'{data_str}-{model_str}-{configs["train_configs"]["train_size"]}{suffix}')


def run_simple_eval(output_dir, data_str, data_key='test', graph_key='graph'):
    
    logger.info('performing simple evaluation')
    graph_est = np.load(os.path.join(output_dir, f'{data_key}_graphs.npy'))
    try:
        graph_true = np.load(os.path.join('data_sim', data_str, f'{graph_key}_{data_key}.npy'))
    except FileNotFoundError:
        try:
            graph_true = np.load(os.path.join('data_sim', data_str, f'{graph_key}.npy'))
            graph_true = np.tile(graph_true[None,:,:],[graph_est.shape[0],1,1])
        except Exception as e:
            logger.error(str(e))
            return
    
    auc_lst = []
    for sample_id in range(graph_est.shape[0]):
        auc = utils_eval.get_auc(graph_true[sample_id], graph_est[sample_id], include_diag=False)
        auc_lst.append(auc)
    auc_mean = pd.DataFrame(auc_lst).mean().to_dict()
    auc_std = pd.DataFrame(auc_lst).std().to_dict()
    
    print('###################')
    print(f'data_key={data_key}; graph_key={graph_key}')
    print(f'averaged across {graph_est.shape[0]} samples: auroc={auc_mean["auroc"]:.2f} ({auc_std["auroc"]:.2f}), auprc={auc_mean["auprc"]:.2f} ({auc_std["auprc"]:.2f})')
    print('###################')
    
    return auc_mean, auc_std


def run_debug_on_train(model, configs, output_dir):
    
    assert output_dir is not None, f'output_dir cannot be none for run_debug_on_train()'
    train_dl_for_test = utils_train.create_one_dataloader(configs['data_configs']['data_str'],
                                            configs['data_configs']['dataset_class'],
                                            configs['train_configs']['batch_size'],
                                            num_workers=4,
                                            mode='train',
                                            size=configs['train_configs']['num_test'],
                                            shuffle=False,
                                            drop_last=False)
    
    preds, graphs = utils_train.model_predict(model, train_dl_for_test)
    np.save(os.path.join(output_dir, 'train_x.npy'), preds)
    np.save(os.path.join(output_dir, 'train_graphs.npy'), graphs)
    
    return



if __name__ == "__main__":
    """
    python -u run_sim.py --data_str=GGM0 --train_size=10000 --gpu=1
    """
    
    logger.info(f'start executing {sys.argv[0]}')
    logger.info(f'python={".".join(map(str,sys.version_info[:3]))}, numpy={np.__version__}, torch={torch.__version__}')
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_str', type=str, required=True, help='string of the dataset name')
    parser.add_argument('--train_size', type=int, required=True, help='number of training samples')
    parser.add_argument('--gpu', type=int, help='GPU id', default=0)
    parser.add_argument('--seed',type=int, help='seed for the run', default=0)
    parser.add_argument('--data_seed',type=int, help='seed for the dataset', default=0)
    parser.add_argument('--output_dir',type=str,help='output folder override, default to None')
    parser.add_argument('--config', type=str, help='override to the config file used; default to None')
    parser.add_argument('--version', type=str, help='version of the config', default=None)
    parser.add_argument('--debug',action='store_true')
    
    args = parser.parse_args()
    
    main(args)
    logger.info(f'end executing {sys.argv[0]}')
    
