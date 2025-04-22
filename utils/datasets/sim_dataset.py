"""
Customized Dataset class for loading both the samples and the underlying graphs
Note that this is only valid for synthetic data where the underlying true graph is known
"""
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class _baseSimDataset(Dataset):
    """
    Base class for Dataset
    """
    def __init__(
        self,
        data_str,
        total_size = None,
        include_diag = False,
    ):
        self.data_folder = os.path.join('data_sim', data_str)
        self.total_size = total_size
        self.include_diag = include_diag
        
    def get_indices_to_extract_(self, num_nodes):
        valid_loc = np.ones([num_nodes, num_nodes])
        if not self.include_diag:
            valid_loc = valid_loc - np.eye(num_nodes)
        indices_to_extract = np.where(valid_loc)
        return indices_to_extract
    
    def __len__(self):
        return self.total_size


class SimDataset(_baseSimDataset):
    
    def __init__(
        self,
        data_str,
        file_type,
        total_size=None,
        include_diag=False
    ):
        super(SimDataset, self).__init__(data_str, total_size, include_diag)
        self.x_path = os.path.join(self.data_folder, f'x_{file_type}.npy')
        self.graph_path = os.path.join(self.data_folder, 'graph.npy')
        
        self.load_x_()
        self.load_graph_()
        
    def load_x_(self):
    
        x = np.load(self.x_path)
        if self.total_size:
            x = x[:self.total_size]
        self.x = x
    
    def load_graph_(self):
        
        graph = np.load(self.graph_path)
        indices_to_extract = self.get_indices_to_extract_(graph.shape[1])
        self.graph = graph[indices_to_extract[0], indices_to_extract[1]]
        
    def __getitem__(self, index):
        
        x = torch.from_numpy(self.x[index]).to(dtype=torch.float32)
        graph = torch.from_numpy(self.graph).to(dtype=torch.float32)
        
        return {'x': x, 'graph': graph}


class SimDatasetWithCovariates(_baseSimDataset):
    
    def __init__(
        self,
        data_str,
        file_type,
        total_size = None,
        include_diag = False
    ):
        super(SimDatasetWithCovariates, self).__init__(data_str, total_size, include_diag)
        self.x_path = os.path.join(self.data_folder, f'x_{file_type}.npy')
        self.z_path = os.path.join(self.data_folder, f'z_{file_type}.npy')
        self.graph_path = os.path.join(self.data_folder, f'graph_{file_type}.npy')
        
        self.load_data_()
        self.load_graph_()

    def load_graph_(self):
        
        graph = np.load(self.graph_path)
        indices_to_extract = self.get_indices_to_extract_(graph.shape[1])
        self.graph = graph[:, indices_to_extract[0], indices_to_extract[1]]
    
    def load_data_(self):
        
        x = np.load(self.x_path)
        z = np.load(self.z_path)
        if self.total_size:
            x = x[:self.total_size]
            z = z[:self.total_size]
        self.x = x
        self.z = z
    
    def __getitem__(self, index):
        
        x = torch.from_numpy(self.x[index]).to(dtype=torch.float32)
        z = torch.from_numpy(self.z[index]).to(dtype=torch.float32)
        graph = torch.from_numpy(self.graph[index]).to(dtype=torch.float32)
        
        return {'x': x, 'z': z, 'graph': graph}
