import numpy as np

from .sim_GGM import GGMBase, GGM3MixBiDiag, GGMMultiMixBlockDiag

SILENCE = True

class NPNTransform():
    
    def __init__(self):
        super(NPNTransform, self).__init__()
        
    @classmethod
    def gaussian_cdf(cls, x, mu=0.05, sigma=0.4):
        dist = norm(loc=mu, scale=sigma)
        return dist.cdf(x)
    
    @classmethod
    def power_transform(cls, x, alpha=3):
        return np.sign(x) * (np.abs(x) ** alpha)
    
    @classmethod
    def sinusoids(cls, x, alpha=3):
        return x + np.sin(alpha * x) / alpha
    
    @classmethod
    def transform_samples(cls, x_from_gaussian, func_type, variance_preserving=True, **kwargs):
        n_samples = x_from_gaussian.shape[0]
        assert func_type in ['power', 'sinusoids', 'gaussian_cdf'], f'unrecognized func_type `{func_type}`'
        if func_type in ['power', 'sinusoids']:
            if not SILENCE:
                print(f'performing (inverse) transformation with func_type={func_type}; alpha={kwargs["alpha"]:.2f}')
            if func_type == 'power':
                x = cls.power_transform(x_from_gaussian, alpha=kwargs['alpha'])
            else: # func_type == 'sinusoids':
                x = cls.sinusoids(x_from_gaussian, alpha=kwargs['alpha'])
        elif func_type == 'gaussian_cdf':
            if not SILENCE:
                print('performing (inverse) transformation with func_type=gaussian_cdf; mu={kwargs["mu"]:.2f}, sigma={kwargs["sigma"]:.2f}')
            x = cls.gaussian_cdf(x_from_gaussian, mu=kwargs['mu'], sigma=kwargs['sigma'])
        else:
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # print(f'@@@@ !![WARNING]: unrecoganized func_type={func_type}; using identity transformation instead')
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # x = x_from_gaussian.copy()
            pass
        
        sd_scaler = kwargs.get('sd_scaler', 1.0)
        if variance_preserving:
            if n_samples > 1:
                sd_scaler_original = sd_scaler
                sd_scaler = np.std(x_from_gaussian, axis=0)/np.std(x, axis=0)
                if not SILENCE:
                    print(f'variance_preserving=True; original scaler={sd_scaler_original} ignored; n_samples={n_samples}; calculated scaler={np.mean(sd_scaler):.2f} ({np.std(sd_scaler):.3f})')
            else:
                if not SILENCE:
                    print(f'n_samples=1; unable to force variance_preserving; using default scaler={sd_scaler:.2f}')
        
        x = sd_scaler * x
        return x
    
class NPNBase(GGMBase):

    def __init__(self, params):
        super(NPNBase, self).__init__(params=params)
        
    def simulate(self, n_samples, func_type, variance_preserving, **kwargs):
        Omega = self.simulate_Omega()
        graph = self.convert_Omega_to_graph(Omega)
        x_from_gaussian = self.simulate_samples_from_Omega(Omega, n_samples)
        x = NPNTransform.transform_samples(x_from_gaussian, func_type, variance_preserving=variance_preserving, **kwargs)
        return {'graph': graph, 'Omega': Omega, 'x': x}


class NPN3MixBiDiag(GGM3MixBiDiag):
    
    def __init__(self, params):
        super(NPN3MixBiDiag, self).__init__(params=params)
        
    def simulate(self, n_samples, func_type, variance_preserving, **kwargs):
        
        Omega_00, Omega_01, Omega_02 = self.simulate_base_Omegas()
        x_lst, z_lst, Omega_lst, graph_lst, regime_lst = [], [], [], [], []
        for _ in range(n_samples):
            x_from_gaussian, z, Omega, regime = self.simulate_one_sample(Omega_00, Omega_01, Omega_02)
            x = NPNTransform.transform_samples(x_from_gaussian, func_type, variance_preserving=variance_preserving, **kwargs)
            x_lst.append(x)
            z_lst.append(z)
            Omega_lst.append(Omega)
            graph_lst.append(self.convert_Omega_to_graph(Omega))
            regime_lst.append(regime)

        Omega = np.stack(Omega_lst, axis=0)
        assert (Omega.ndim==3) and (Omega.shape[0] == n_samples) and (Omega.shape[1] == self.n_nodes) and (Omega.shape[2] == self.n_nodes), f'Omega.shape={Omega.shape}'
        graph = np.stack(graph_lst, axis=0)
        regime = np.array(regime_lst)

        x = np.concatenate(x_lst, axis=0)
        assert (x.ndim == 2) and (x.shape[0] == n_samples) and (x.shape[1] == self.n_nodes), f'x.shape={x.shape}'
        z = np.stack(z_lst, axis=0)
        assert (z.ndim == 2) and (z.shape[0] == n_samples), f'z.shape={z.shape}'

        return {'graph': graph, 'Omega': Omega, 'x': x, 'z': z, 'regime': regime}


class NPNMultiMixBlockDiag(GGMMultiMixBlockDiag):
    
    def __init__(self, params):
        super(NPNMultiMixBlockDiag, self).__init__(params=params)
    
    def simulate(self, n_samples, func_type, variance_preserving, **kwargs):
        
        Omega_00, Omega_01, Omega_02 = self.simulate_base_Omegas()
        x_lst, z_lst, Omega_lst, graph_lst, regime_lst = [], [], [], [], []
        for _ in range(n_samples):
            x_from_gaussian, z, Omega, regime = self.simulate_one_sample(Omega_00, Omega_01, Omega_02)
            x = NPNTransform.transform_samples(x_from_gaussian, func_type, variance_preserving=variance_preserving, **kwargs)
            x_lst.append(x)
            z_lst.append(z)
            Omega_lst.append(Omega)
            graph_lst.append(self.convert_Omega_to_graph(Omega))
            regime_lst.append(regime)

        Omega = np.stack(Omega_lst, axis=0)
        assert (Omega.ndim==3) and (Omega.shape[0] == n_samples) and (Omega.shape[1] == self.n_nodes) and (Omega.shape[2] == self.n_nodes), f'Omega.shape={Omega.shape}'
        graph = np.stack(graph_lst, axis=0)
        regime = np.array(regime_lst)

        x = np.concatenate(x_lst, axis=0)
        assert (x.ndim == 2) and (x.shape[0] == n_samples) and (x.shape[1] == self.n_nodes), f'x.shape={x.shape}'
        z = np.concatenate(z_lst, axis=0)
        assert (z.ndim == 2) and (z.shape[0] == n_samples), f'z.shape={z.shape}'

        return {'graph': graph, 'Omega': Omega, 'x': x, 'z': z, 'regime': regime}
