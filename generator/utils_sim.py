import importlib
import os
import pickle
import random
import shutil

import numpy as np
import yaml

from utils import utils_logging

logger = utils_logging.get_logger()

def setup_(args):

    args.config = args.config or os.path.join('configs', '_synthetic_.yaml')
    
    logger.info(f'initial setup; dataset={args.data_str}; seed={args.data_seed}; config_file={args.config}')
    with open(args.config) as f:
        configs = yaml.safe_load(f)[args.data_str]

    args.folder_name = os.path.join('data_sim', f'{args.data_str}_seed{args.data_seed}')
    if os.path.exists(args.folder_name):
        logger.warning(f'folder={args.folder_name} existed; deleting it')
        shutil.rmtree(args.folder_name)
    os.makedirs(args.folder_name)
    logger.info(f'folder={args.folder_name} created')
        
    ## set seed
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    
    ## retrieve simulators
    available_simulators = importlib.import_module('generator.simulator')
    
    try:
        logger.info(f'retrieving simulator={configs["simulator"]}')
        SimulatorObj = getattr(available_simulators, configs['simulator'])
        simulator = SimulatorObj(configs)
        assert (simulator is not None)
        logger.info(f'simulator={simulator.__class__.__name__} retrieved')
    except AttributeError:
        shutil.rmtree(args.folder_name)
        raise ValueError(f'simulator {configs["simulator_class"]} not found')
    
    return configs, simulator


def parse_train_val_test_indices(configs):
    
    return {'train': range(configs['n_train']),
             'val': range(configs['n_train'], configs['n_train']+configs['n_val']),
             'test': range(configs['n_train']+configs['n_val'], configs['n_samples'])
             }


def save_by_mode(data_to_save, data_key, mode_key, folder_name, file_type='.npy'):

    save_path = os.path.join(folder_name, f'{data_key}_{mode_key}' + file_type)
    if file_type == '.npy':
        np.save(save_path, data_to_save)
    elif file_type == '.pickle':
        with open(save_path, 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return


def save_samples(configs, output_from_simulation, folder_name):

    logger.info('stage: saving samples')
    indices = parse_train_val_test_indices(configs)
    
    data_keys = ['x', 'z']
    for key in data_keys:
        if key not in output_from_simulation:
            logger.warning(f'key `{key}` is not part of the simulation output; skipped')
            continue
        samples = output_from_simulation[key]
        for mode in ['train', 'val', 'test']:
            data_to_save = samples[indices[mode]]
            save_by_mode(data_to_save, data_key=key, mode_key=mode, folder_name=folder_name, file_type='.npy')
        logger.info(f'key={key} is saved as `{folder_name}/{key}_(train,val,test).npy`; shape={samples.shape}')

    return


def save_params(configs, output_from_simulation, folder_name):

    logger.info('stage: saving params')
    
    if configs['simulator'].startswith('GGM') or configs['simulator'].startswith('NPN'):
        param_keys = ['graph', 'Omega', 'regime']
    elif configs['simulator'].startswith('LinearSEM') or configs['simulator'].startswith('HermiteSEM'):
        param_keys = ['graph', 'A', 'Omega', 'symSkel', 'regime']
    else:
        raise ValueError(f'Currently do not know how to save parameters for simulator={configs["simulator"]}')
    
    if output_from_simulation['graph'].ndim == 2:
        for key in param_keys:
            param = output_from_simulation[key]
            np.save(os.path.join(folder_name, f'{key}.npy'), param)
            logger.info(f'key={key} is saved as `{folder_name}/{key}.npy`; shape={param.shape}')
    else:
        indices = parse_train_val_test_indices(configs)
        for key in param_keys:
            param = output_from_simulation[key]
            for mode in ['train', 'val', 'test']:
                param_to_save = param[indices[mode]]
                save_by_mode(param_to_save, data_key=key, mode_key=mode, folder_name=folder_name, file_type='.npy')
            logger.info(f'key={key} is saved as `{folder_name}/{key}_(train,val,test).npy`; shape={param.shape}')

    return


def debug(output_from_simulation):
    
    try:
        Omega = output_from_simulation['Omega']
        if Omega.ndim == 2:
            eigenvalues, _ = np.linalg.eig(Omega)
            cn = np.max(eigenvalues)/np.min(eigenvalues)
        elif Omega.ndim == 3:
            cns = []
            for i in range(Omega.shape[0]):
                eigenvalues, _ = np.linalg.eig(Omega[i])
                cn = np.max(eigenvalues)/np.min(eigenvalues)
                cns.append(cn)
            cn = np.mean(cns)
        logger.debug(f'condition number for Omega={cn:.2f}')
    except Exception as e:
        logger.error(str(e))
    
    return
