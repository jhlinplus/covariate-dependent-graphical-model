## modified based on https://stackoverflow.com/questions/71891964/how-to-load-early-stopping-counter-in-pytorch
import os

import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(
        self,
        patience=5,
        min_delta=1.0e-4,
        ckpt_dir='ckpt',
        verbose=False,
    ):
        """
        Args:
        - patience (int): How long to wait after last time validation loss improved.
        - min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
        - path (str): directory for the checkpoint to be saved to.
        - verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_min = np.Inf
        
    def __call__(self, metric_val, model_state_dict, save_ckpt=False):

        if self.best_score is None:
            self.best_score = metric_val
            self.save_checkpoint(metric_val, model_state_dict)
        elif metric_val >= self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            if self.verbose:
                print(f'** EarlyStopping counter: {self.counter} out of {self.patience}; (best_score={self.best_score:.5f}, current_score={metric_val:.5f}')
        else:
            if save_ckpt:
                if self.verbose:
                    print(f'monitored metric decreased ({self.best_score:.5f} ==> {metric_val:.5f}). saving model ...')
                self.save_checkpoint(metric, model_state_dict)
            self.counter = 0
            self.best_score = metric_val
            
    def save_checkpoint(self, metric, model_state_dict):
        torch.save({'model_state_dict': model_state_dict}, os.path.join(self.ckpt_dir, "ckpt.pth"))
        
