import numpy as np
from scipy.special import expit

from .rbf import RBF

class OmegaSimulator():
    """
    simulator with a number of utility functions for generating Omega from a Gaussian Graphical Model
    """
    def __init__(self, n_nodes):
        super(OmegaSimulator, self).__init__()
        self.n_nodes = n_nodes
        
    def bump_diag(self, Omega, target_cn):
        q = Omega.shape[0]
        eigenvalues, _ = np.linalg.eig(Omega)
        ## bring it above zero
        if np.min(eigenvalues) < 0:
            Omega = Omega + np.eye(q) * (1.0e-3 + np.abs(np.min(eigenvalues)))
            eigenvalues, _ = np.linalg.eig(Omega)
        cn = np.max(eigenvalues)/np.min(eigenvalues)
        while cn > target_cn:
            Omega = Omega + 0.005*np.eye(q)
            eignew, _ = np.linalg.eig(Omega)
            cn = np.max(eignew)/np.min(eignew)
        return Omega
    
    def assert_positive_definite(self, Omega):
        eigenvalues, _ = np.linalg.eig(Omega)
        assert np.min(eigenvalues) > 0
    
    def generate_Omega_random(self, condition_number, sparsity, sig_low, sig_high):
        Omega = np.zeros((self.n_nodes,self.n_nodes))
        if sparsity is None:
            sparsity = 2.0/self.n_nodes
        for i in range(self.n_nodes-1):
            for j in range(i,self.n_nodes):
                Omega[i,j] = (np.random.binomial(1,sparsity,1) * np.random.choice([-1,1],size=1) * np.random.uniform(low=sig_low,high=sig_high,size=1))[0]
                Omega[j,i] = Omega[i,j].copy()
        Omega = self.bump_diag(Omega, condition_number)
        return Omega
    
    def generate_Omega_bidiag(self, condition_number, sig, offset=1):
        Omega = np.zeros((self.n_nodes,self.n_nodes))
        for i in range(self.n_nodes-1):
            for j in range(i,self.n_nodes):
                Omega[i,j] = sig if j-i==offset else 0
                Omega[j,i] = Omega[i,j].copy()
        Omega = self.bump_diag(Omega, condition_number)
        return Omega

    def generate_Omega_tridiag(self, condition_number, sig_low, sig_high):
        Omega = np.zeros((self.n_nodes,self.n_nodes))
        for i in range(self.n_nodes-1):
            for j in range(i,self.n_nodes):
                Omega[i,j] = sig_high if j-i==1 else (sig_low if j-i == 2 else 0)
                Omega[j,i] = Omega[i,j].copy()
        Omega = self.bump_diag(Omega, condition_number)
        return Omega
        
    def generate_Omega_blockdiag(self, n_blocks, sparsity, condition_number, sig_low, sig_high):
        
        self.n_nodes % n_blocks == 0
        block_size = self.n_nodes // n_blocks
        Omega = np.zeros((self.n_nodes,self.n_nodes))
        for block_idx in range(n_blocks):
            start, end = block_idx * block_size, (block_idx+1) * block_size
            for row_idx in range(start, end):
                for col_idx in range(row_idx+1, end):
                    Omega[row_idx, col_idx] = (np.random.binomial(1,sparsity,1) * np.random.choice([-1,1],size=1) * np.random.uniform(low=sig_low,high=sig_high,size=1))[0]
                    Omega[col_idx, row_idx] = Omega[row_idx, col_idx].copy()
        
        Omega = self.bump_diag(Omega, condition_number)
        return Omega
    
    @classmethod
    def convert_Omega_to_graph(cls, Omega):
        """
        infer the betas from Omega; e.g., eqn (4) in https://stat.ethz.ch/Manuscripts/buhlmann/gelato.pdf
        """
        OmegaDiagInv = np.diag(1.0/np.diag(Omega))
        graph = (-1) * OmegaDiagInv @ Omega
        np.fill_diagonal(graph, 0)
        return graph


class GGMBase(OmegaSimulator):
    """
    base case simulator does not consider external covariates
    """
    def __init__(self, params):
        super(GGMBase, self).__init__(n_nodes=params['n_nodes'])
        self.params = params
    
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        if params['graph_type'] in ['random','tridiag']:
            assert ('sig_low' in params) and ('sig_high' in params)
        elif params['graph_type'] == 'bidiag':
            assert 'sig' in params
            params['offset'] = params.get('offset', 1)
        else:
            raise ValueError('unrecognized graph_type {params["graph_type"]}')
        self._params = params

    def simulate_Omega(self):
        if self.params['graph_type'] == 'random':
            Omega = self.generate_Omega_random(self.params['cond_number'], self.params.get('sparsity', None), self.params['sig_low'], self.params['sig_high'])
        elif self.params['graph_type'] == 'bidiag':
            Omega = self.generate_Omega_bidiag(self.params['cond_number'], self.params['sig'], offset=self.params['offset'])
        elif self.params['graph_type'] == 'tridiag':
            Omega = self.generate_Omega_tridiag(self.params['cond_number'], self.params['sig_low'], self.params['sig_high'])    
        else:
            raise Exception
        return Omega

    def simulate_samples_from_Omega(self, Omega, n_samples):
        x = np.random.multivariate_normal(np.zeros((Omega.shape[0],)),np.linalg.inv(Omega), size=n_samples)
        return x
    
    def simulate(self, n_samples):
        ## get Omega and samples
        Omega = self.simulate_Omega()
        x = self.simulate_samples_from_Omega(Omega, n_samples)
        ## convert omega to graph whose non-zero corresponds to the betas
        graph = self.convert_Omega_to_graph(Omega)
        return {'graph': graph, 'Omega': Omega, 'x': x}


class GGM3MixBiDiag(OmegaSimulator):
    """
    simulator class for the following setting
    (0) base Omegas. Omega^0, Omega^1, Omega^3, bidiag with offset 1, 2 and 3, respectively
    (1) for each sample (x_i, z_i):
        - sample a 2-dim z_i:=(z_i0, z_i1) from unif(0,1); Omega_i = (z_i1**2) * Omega^k + (1-z_i1**2) * Omega^j; where the choice of k,j depends on the value of z_i0
        - sample x_i
    """
    def __init__(self, params):
        super(GGM3MixBiDiag, self).__init__(params['n_nodes'])
        self.params = params
    
    def simulate_base_Omegas(self):
        Omega_00 = self.generate_Omega_bidiag(self.params['cond_number'],self.params['sig'], offset=1)
        Omega_01 = self.generate_Omega_bidiag(self.params['cond_number'],self.params['sig'], offset=2)
        Omega_02 = self.generate_Omega_bidiag(self.params['cond_number'],self.params['sig'], offset=3)
        return Omega_00, Omega_01, Omega_02

    def simulate_one_sample(self, Omega_00, Omega_01, Omega_02):
        
        z = np.random.uniform(low=0.0,high=1.0,size=(2,))
        
        if z[1] < 1.0/3:
            Omega_a, Omega_b = Omega_00, Omega_01
            regime_id = 1
        elif z[1] < 2.0/3:
            Omega_a, Omega_b = Omega_00, Omega_02
            regime_id = 2
        else:
            Omega_a, Omega_b = Omega_01, Omega_02
            regime_id = 3
        
        mixing_pct = min(0.75, max(0.25, z[0]))
        Omega = mixing_pct * Omega_a + (1.0-mixing_pct) * Omega_b
        x = np.random.multivariate_normal(np.zeros((Omega.shape[0],)),np.linalg.inv(Omega), size=(1,))
        return x, z, Omega, regime_id


    def simulate(self, n_samples):
        
        Omega_00, Omega_01, Omega_02 = self.simulate_base_Omegas()
        x_lst, z_lst, regime_lst = [], [], []
        Omega_lst, graph_lst = [], []
        for _ in range(n_samples):
            x, z, Omega, regime = self.simulate_one_sample(Omega_00, Omega_01, Omega_02)
            x_lst.append(x)
            z_lst.append(z)
            regime_lst.append(regime)
            Omega_lst.append(Omega)
            graph_lst.append(self.convert_Omega_to_graph(Omega))

        Omega = np.stack(Omega_lst, axis=0)
        assert (Omega.ndim==3) and (Omega.shape[0] == n_samples) and (Omega.shape[1] == self.n_nodes) and (Omega.shape[2] == self.n_nodes), f'Omega.shape={Omega.shape}'
        graph = np.stack(graph_lst, axis=0)
        regime = np.array(regime_lst)

        x = np.concatenate(x_lst, axis=0)
        assert (x.ndim == 2) and (x.shape[0] == n_samples) and (x.shape[1] == self.n_nodes), f'x.shape={x.shape}'
        z = np.stack(z_lst, axis=0)
        assert (z.ndim == 2) and (z.shape[0] == n_samples), f'z.shape={z.shape}'

        return {'graph': graph, 'Omega': Omega, 'x': x, 'z': z, 'regime': regime}


class GGMMultiMixBlockDiag(OmegaSimulator):
    
    def __init__(self, params):
        super(GGMMultiMixBlockDiag, self).__init__(params['n_nodes'])
        self.params = params
        self.rbf = RBF(input_dim = self.params['dim_z'],
                       K = self.params['rbf_K'],
                       c_range = self.params['rbf_c_range'],
                       beta_range = self.params['rbf_beta_range'],
                       w_range = self.params['rbf_w_range'])
    
    def zero_out(self, Omega, block_to_retain_start, block_to_retain_end):
        
        Omega = Omega.copy()
        for i in range(self.n_nodes-1):
            for j in range(i+1, self.n_nodes):
                if block_to_retain_start <= i < block_to_retain_end and block_to_retain_start <= j < block_to_retain_end:
                    continue
                else:
                    Omega[i,j], Omega[j,i] = 0, 0
        return Omega
    
    def simulate_base_Omegas(self):
    
        block_size = self.n_nodes // self.params['n_blocks']
        Omega_00 = self.generate_Omega_blockdiag(self.params['n_blocks'], self.params['sparsity'], self.params['cond_number'], self.params['sig_low'], self.params['sig_high'])
        Omega_00 = self.zero_out(Omega_00, 0, block_size)
        
        Omega_01 = self.generate_Omega_blockdiag(self.params['n_blocks'], self.params['sparsity'], self.params['cond_number'], self.params['sig_low'], self.params['sig_high'])
        Omega_01 = self.zero_out(Omega_01, block_size, 2*block_size)
        
        Omega_02 = self.generate_Omega_blockdiag(self.params['n_blocks'], self.params['sparsity'], self.params['cond_number'], self.params['sig_low'], self.params['sig_high'])
        Omega_02 = self.zero_out(Omega_02, 2*block_size, 3*block_size)
        
        return Omega_00, Omega_01, Omega_02
    

    def simulate_one_sample(self, Omega_00, Omega_01, Omega_02):
        
        z = np.random.normal(size=(self.params['dim_z'],)).reshape(1,-1)
        z_rbf = self.rbf(z)
        z_rbf_sigmoid = expit(z_rbf)
        
        if z_rbf_sigmoid > 0.9 or z_rbf_sigmoid < 0.1:
            mix_pct = min(0.75, max(z_rbf_sigmoid, 0.25))
            Omega = mix_pct * Omega_00 + (1-mix_pct) * Omega_02
            regime_id = 1
        else:
            mix_01 = 0.5
            mix_00 = min(0.35, max(z_rbf_sigmoid, 0.15))
            mix_02 = 0.5 - mix_00
            Omega = mix_00 * Omega_00 + mix_01 * Omega_01 + mix_02 * Omega_02
            regime_id = 2
            
        x = np.random.multivariate_normal(np.zeros((Omega.shape[0],)),np.linalg.inv(Omega), size=(1,))
        return x, z, Omega, regime_id


    def simulate(self, n_samples):
        
        Omega_00, Omega_01, Omega_02 = self.simulate_base_Omegas()
        x_lst, z_lst, regime_lst = [], [], []
        Omega_lst, graph_lst = [], []
        for _ in range(n_samples):
            x, z, Omega, regime = self.simulate_one_sample(Omega_00, Omega_01, Omega_02)
            x_lst.append(x)
            z_lst.append(z)
            regime_lst.append(regime)
            Omega_lst.append(Omega)
            graph_lst.append(self.convert_Omega_to_graph(Omega))

        Omega = np.stack(Omega_lst, axis=0)
        assert (Omega.ndim==3) and (Omega.shape[0] == n_samples) and (Omega.shape[1] == self.n_nodes) and (Omega.shape[2] == self.n_nodes), f'Omega.shape={Omega.shape}'
        graph = np.stack(graph_lst, axis=0)
        regime = np.array(regime_lst)

        x = np.concatenate(x_lst, axis=0)
        assert (x.ndim == 2) and (x.shape[0] == n_samples) and (x.shape[1] == self.n_nodes), f'x.shape={x.shape}'
        z = np.concatenate(z_lst, axis=0)
        assert (z.ndim == 2) and (z.shape[0] == n_samples), f'z.shape={z.shape}'

        return {'graph': graph, 'Omega': Omega, 'x': x, 'z': z, 'regime': regime}
