import numpy as np
import torch
import torch.nn as nn

from .modules import MLP, ResidualBlock

class _baseGM(nn.Module):

    def __init__(self, num_nodes: int, include_diag: bool=False):

        super(_baseGM, self).__init__()
        self.num_nodes = num_nodes
        self.include_diag = include_diag
        self.create_receiver_sender_indices()

    def create_receiver_sender_indices(self):
        edge_indices = np.ones([self.num_nodes, self.num_nodes])
        if not self.include_diag:
            edge_indices = np.ones([self.num_nodes, self.num_nodes]) - np.eye(self.num_nodes)
        self.receivers, self.senders = np.where(edge_indices)
        return
    
    def count_num_params(self, module_name: str=None):
        module = self if module_name is None else getattr(self, module_name)
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    def organize_data_self_rm(self, data: torch.tensor):
        """
        create regressors for each node
        argv:
        - data: [batch_size, num_nodes]
        return:
        - data_rm: [batch_size, num_nodes, num_nodes-1]; note that [*, i, :] hosts the regressors (i.e., all nodes but i) when i is the response node
        """
        mask = (1 - torch.eye(self.num_nodes, device=data.device)).bool()
        data_unmasked = data.unsqueeze(1).expand(data.shape[0], data.shape[1], data.shape[1])
        data_rm = torch.masked_select(data_unmasked, mask).reshape(data.shape[0], data.shape[1], -1).contiguous()
        return data_rm
    
    def unflatten_graph(self, graph: torch.tensor):
        """
        reshape graph [batch_size, num_edges] -> [batch_size, num_nodes, num_nodes]
        """
        graph_reshape = torch.zeros(graph.shape[0], self.num_nodes, self.num_nodes, device=graph.device)
        graph_reshape[:,self.receivers, self.senders] = graph
        return graph_reshape


class dnnCGM(_baseGM):
    """
    DNN [C]ovariate [G]raphical [M]model
    each response node i is modeled as follows: x_i = \sum_{j \neq i} beta^i_j(z) gamma^i_j(x_j) + noise
    """
    def __init__(self, configs):
        super(dnnCGM, self).__init__(configs['num_nodes'], include_diag=False)
        self.configs = configs
        self.dummy = nn.Parameter(torch.empty(0))
        self.num_nodes = self.configs['num_nodes']
        self.num_covariates = self.configs['num_covariates']
        if self.configs['beta_module_name'] == 'ResidualBlock':
            self.beta_network = ResidualBlock(input_dim = self.num_covariates,
                                              output_dim = self.num_nodes*(self.num_nodes-1),
                                              hidden_dim = self.configs['beta_hidden_dims'],
                                              dropout_rate = self.configs['beta_dropout'],
                                              batch_norm = self.configs['beta_batch_norm'],
                                              bias = self.configs['beta_bias'])
        else:
            self.beta_network = MLP(input_dim = self.num_covariates,
                                    output_dim = self.num_nodes*(self.num_nodes-1),
                                    hidden_dims = self.configs['beta_hidden_dims'],
                                    dropout_rate = self.configs['beta_dropout'],
                                    batch_norm = self.configs['beta_batch_norm'],
                                    bias = self.configs['beta_bias'])
            
    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        configs = configs.copy()
        configs['beta_module_name'] = configs.get('beta_module_name', 'ResidualBlock')
        assert configs['beta_module_name'] in ['ResidualBlock', 'MLP'], \
            f'unrecognized `beta_module_name`=configs["beta_module_name"]; must be either `ResidualBlock` or `MLP`'
        configs['beta_batch_norm'] = configs.get('beta_batch_norm', True)
        configs['beta_bias'] = configs.get('beta_bias', True)
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
        ## input to the beta function is z of size [batch_size, num_covariates]
        ## output is the beta_j^i for all i=1,...,p; j\neq i of size [batch_size, p(p-1)]
        graph = self.beta_network(z)
        ## simple reshape
        beta = graph.reshape(batch_size, self.num_nodes, self.num_nodes-1) ## beta[*, i, :] hosts the beta^i_j's, j\neq i
        ## handle the "gamma" part; note that since we are using identity, it suffices to do some basic data manipulation
        h_output = self.organize_data_self_rm(x)
        x_pred = (h_output * beta).sum(axis=-1)
        return x_pred, graph
