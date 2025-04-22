import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    a simple feed-forward network
    consisting of a sequence of linear -> activation -> dropout layers
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int]|int|None,
        dropout_rate: float=0.1,
        batch_norm: bool=True,
        bias: bool=True,
    ) -> None:
        
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        ## linear -> (batch_norm) -> relu -> dropout -> linear
        if hidden_dims is None:
            self.feed_forward = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]
                
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout_rate))
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
            self.feed_forward = nn.Sequential(*layers)
        
    def forward(self, x):

        return self.feed_forward(x)


class ResidualBlock(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout_rate: float=0.1,
        batch_norm: bool=True,
        bias: bool=True,
    ) -> None:
    
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
    
        self.mlp1 = MLP(input_dim = self.input_dim,
                        hidden_dims = self.hidden_dim,
                        output_dim = self.hidden_dim//2,
                        dropout_rate = dropout_rate,
                        batch_norm = batch_norm,
                        bias=bias)
        
        self.mlp2 = MLP(input_dim = self.input_dim + self.hidden_dim//2,
                        hidden_dims = self.hidden_dim,
                        output_dim = self.output_dim,
                        dropout_rate = dropout_rate,
                        batch_norm = False,
                        bias = bias)
    
    
    def forward(self, x):
        
        y = self.mlp1(x)
        xy = torch.cat([x, y], axis=-1)
        out = self.mlp2(xy)
        
        return out
