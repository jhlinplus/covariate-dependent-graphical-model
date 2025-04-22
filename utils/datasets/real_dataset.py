import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class RealDatasetWithCovariates(Dataset):
    
    def __init__(self, x, z):
        super(RealDatasetWithCovariates, self).__init__()
        self.x = x
        self.z = z
        self.sample_size = x.shape[0]
    
    def __len__(self):
        return self.sample_size
        
    def __getitem__(self, index):
        
        x = torch.from_numpy(self.x[index]).to(dtype=torch.float32)
        z = torch.from_numpy(self.z[index]).to(dtype=torch.float32)
        
        return {'x': x, 'z': z}
