#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path
import os
import random
import shutil
import sys

_ROOTDIR_ = str(Path(__file__).resolve().parents[1])
sys.path.append(_ROOTDIR_)
os.chdir(_ROOTDIR_)

import numpy as np
import yaml

from generator import utils_sim
from utils import utils_logging

logger = utils_logging.get_logger()


def main(args):

    configs, simulator = utils_sim.setup_(args)
    
    ## simulate
    logger.info(f'stage: producing synthetic data')
    if configs['func_type'] == 'gaussian_cdf':
        output_from_simulation = simulator.simulate(n_samples = configs['n_samples'], func_type = configs['func_type'], variance_preserving=True, mu=configs['mu'], sigma=configs['sigma'])
    elif configs['func_type'] in ['power', 'sinusoids']:
        output_from_simulation = simulator.simulate(n_samples = configs['n_samples'], func_type = configs['func_type'], variance_preserving=True, alpha=configs['alpha'])
    else:
        raise ValueError(f'invalid func_type={configs["func_type"]}; choose between `gaussian_cdf`, `power_transformation` or `sinusoids`')
    
    ## save down xs and zs, Omega and graph
    utils_sim.save_samples(configs, output_from_simulation, args.folder_name)
    utils_sim.save_params(configs, output_from_simulation, args.folder_name)
    
    if args.debug:
        utils_sim.debug(output_from_simulation)
    
    return 0


if __name__ == "__main__":
    
    logger.info(f'start executing {sys.argv[0]}')
    logger.info(f'python={".".join(map(str,sys.version_info[:3]))}, numpy={np.__version__}')
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_str', type=str, help='name for dataset to be generated',default='GGM0')
    parser.add_argument('--data_seed', type=int, help='seed value',default=0)
    parser.add_argument('--config', type=str, help='config file override, default to None')
    parser.add_argument('--debug', action='store_true')
    
    global args
    args = parser.parse_args()
    
    main(args)
    logger.info(f'end executing {sys.argv[0]}')
    
