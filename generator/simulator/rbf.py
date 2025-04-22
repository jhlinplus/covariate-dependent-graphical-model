import numpy as np

class RBF():
    """
    conduct the mapping x -> f(x) where f is parametrized by a radial basis function network: R^d -> R
    - f(x) = \sum_{k=1}^K w_k \exp{ -beta_k * ||x - c_k||^2 }
    - K is the number of hidden layer neurons (hidden_dim)
    """
    def __init__(
        self,
        input_dim,
        K = 3,
        c_range = [-1, 1],
        beta_range = [0.01, 0.02],
        w_range = [-10, 10]
    ):
        self.input_dim = input_dim
        self.output_dim = 1
        self.K = K
        self.c_range = c_range
        self.beta_range = beta_range
        self.w_range = w_range
        
        self.assign_params_()
        
    def assign_params_(self):
        """
        assign parameters
        """
        ## centers (c_k's)
        self.cs = np.random.uniform(low=self.c_range[0], high=self.c_range[1], size=(self.K, self.input_dim))
        ## scales (beta_k's)
        self.betas = np.random.uniform(low=self.beta_range[0], high=self.beta_range[1], size=(self.K,))
        ## weights to the output layer (i.e., the w_k's)
        self.weights = np.random.uniform(low=self.w_range[0], high=self.w_range[1], size=(self.output_dim, self.K))
    
    def kernelize(self, x):
        """
        convert a "batch" of x through the kernels; 
        Argv:
        - x.shape = [B, input_dim]
        Return:
        - y.shape = [B, K]
        """
        assert x.shape[1] == self.input_dim
        if x.ndim == 1:
            x = x[:,None]
        kernel_out = np.zeros((x.shape[0], self.K))
        for k in range(self.K):
            kernel_out[:,k] = np.exp(- self.betas[k] * np.linalg.norm(x - self.cs[k,:],axis=1) ** 2)
        
        return kernel_out
    
    def __call__(self,x):
        """
        convert a "batch" of x -> f(x)
        Argv:
        - x.shape = [B, input_dim]
        Return:
        - y.shape = [B, 1]
        """
        kernel_out = self.kernelize(x)
        fx = np.matmul(kernel_out, self.weights.transpose())
        
        return fx.squeeze(-1)
