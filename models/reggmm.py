"""
This script implements RegGMM (Zhang & Li, 2023 JASA)
where the dependency on the covariates are assumed linear (eqn 6 in https://arxiv.org/pdf/2011.05245)
"""
import numpy as np
import torch
import torch.nn as nn

from .networks import _baseGM

class RegGMM(_baseGM):
    
    def __init__(self, configs):
        super(RegGMM, self).__init__(configs['num_nodes'], include_diag=False)
        self.configs = configs
        self.dummy = nn.Parameter(torch.empty(0))
        self.num_nodes = self.configs['num_nodes']
        self.num_covariates = self.configs['num_covariates']
        self.beta_network = nn.Linear(self.num_covariates,
                                      self.num_nodes*(self.num_nodes-1),
                                      bias=self.configs['bias'])

    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        configs = configs.copy()
        configs.setdefault('bias', True)
        self._configs = configs

    def print_summary(self):
        print('=' * 20)
        print(f'total number of parameters: {self.count_num_params():,}')
        print('=' * 20)

    def forward(self, x, z):
        """
        Argv:
        - x: [batch_size, num_nodes]
        - z: [batch_size, num_covariates]
        Return:
        - x_pred: predicted values based on the node-wise regression [batch_size, num_nodes]
        - graph: estimated beta coefficient, corresponding to the (scaled) graphical entries [batch_size, num_edges]
        """
        batch_size = x.shape[0]
        graph = self.beta_network(z)
        beta = graph.reshape(batch_size, self.num_nodes, self.num_nodes-1)
        x_reorg = self.organize_data_self_rm(x)
        x_pred = (x_reorg * beta).sum(axis=-1)
        
        return x_pred, graph
