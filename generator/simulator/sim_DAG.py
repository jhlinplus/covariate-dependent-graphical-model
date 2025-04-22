import time

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from .hermite_functions import hermite_functions, hermite_polynomials, hermite_function_linear_coefs, hermite_polynomial_linear_coefs

DEBUG = False


class SkeletonSimulator():
    """
    base class for simulating the skeleton of a DAG and getting its corresponding moral graph
    """
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        
    def generate_DAG_skeleton(self, graph_type, **kwargs):
        """
        simulate the skeleton of a DAG which is a lower or upper triangular adjacency matrix
        NOTE1 the lower-triangular structure automatically gives the topological ordring
        NOTE2: in our adj mtx representation, A[i,j]=1 corresponds to node j being a parent of node i
        NOTE3: nx treats A[i,j]=1 as node i being a parent of node j; so be careful with the transpose operations
        
        Argv (required):
        - graph_type: str, choose among erdos-renyi, barabasi-albert or chain-x where x is an integer
        Argv (kwargs):
        - sparsity: float that dictates the density level; needed for erdos-renyi and barabasi-albert
        - n_children_low/high: number of children in the form of lower and upper bounds; needed for tree
        Returns:
        - skeleton: np.array denoting the skeleton of the DAG
        """
        if graph_type == 'random': ## erdos-renyi
            skeleton = np.zeros((self.n_nodes, self.n_nodes))
            sparsity = kwargs['sparsity']
            for j in range(self.n_nodes-1):
                skeleton[(j+1):,j] = np.random.binomial(1,sparsity,size=(self.n_nodes-j-1,))
        elif graph_type == 'barabasi-albert':
            sparsity = kwargs['sparsity']
            m = int(round(sparsity*(self.n_nodes-1)/2))
            G0 = nx.barabasi_albert_graph(self.n_nodes, m, seed=None, initial_graph=None)
            skeleton = np.triu(nx.to_numpy_array(G0),k=1).transpose()
        elif graph_type == 'chain':
            chain_size = kwargs['chain_size']
            skeleton = np.zeros((self.n_nodes,self.n_nodes))
            for j in range(self.n_nodes):
                for k in range(j+1,min(j+chain_size+1,self.n_nodes)):
                    skeleton[k,j] = 1
        elif graph_type == 'tree':
            ## lower and upper bound for the number of children
            n_children_low, n_children_high = kwargs['n_children_low'], kwargs['n_children_high']
            skeleton = np.zeros((self.n_nodes, self.n_nodes))
            ## generating the tree
            node_queue, slot_remaining = [0], self.n_nodes - 1
            while slot_remaining > 0:
                parent_node_id = node_queue.pop(0)
                n_children = min(np.random.randint(n_children_low, n_children_high+1, 1)[0], slot_remaining)
                slot_start_idx = self.n_nodes - slot_remaining
                children_node_ids = list(range(slot_start_idx, slot_start_idx + n_children))
                for child_id in children_node_ids:
                    skeleton[child_id, parent_node_id] = 1
                node_queue.extend(children_node_ids)
                slot_remaining -= n_children
            ## validate
            DAG = nx.from_numpy_array(skeleton.transpose(), create_using=nx.DiGraph)
            assert nx.is_tree(DAG)
        else:
            raise ValueError(f'unrecognized graph_type={graph_type}')
        
        return skeleton
    
    @classmethod
    def assert_triangular(cls, mtx):
        assert np.allclose(mtx, np.tril(mtx)) or np.allclose(mtx, np.triu(mtx))
    
    @classmethod
    def get_moral_graph(cls, skeleton):
        """
        get model graph from that of the DAG
        NOTE that in the DAG skeleton specification, A[i,j]=1 means node j being a parent of node i
        Argv:
        - skeleton: np.array, adj_mtx/skeleton of the DAG
        Return:
        - skeleton_moral: np.array, adj_mtx/skeleton of the moralized graph
        """
        DAG = nx.from_numpy_array(skeleton.transpose(), create_using=nx.DiGraph)
        DAG_moral = nx.moral_graph(DAG)
        skeleton_moral = nx.to_numpy_array(DAG_moral).transpose()
        
        return skeleton_moral
    
    @classmethod
    def get_moral_graph_magnitude(cls, A, Omega_epsilon=None):
        """
        - A: np.array, adj_mtx of the DAG; assuming x = Ax + noise, either exact or approximately equal
        - Omega_epsilon: noise covariance of the DAG representation
        """
        ## Sigma is given in equation (4) in https://jmlr.csail.mit.edu/papers/volume15/loh14a/loh14a.pdf
        ## note that in their notation, B^T == our A
        ## moral_graph is given by the inv covariance (I-A^T) Omega_epsilon^{-1} (I-A^T)^T
        if Omega_epsilon is None:
            Omega_epsilon = np.identity(A.shape[0])
            
        Omega_epsilon_inv = np.linalg.inv(Omega_epsilon)
        I_minus_AT = np.identity(A.shape[0]) - A.transpose()
        magnitude_moral = I_minus_AT @ Omega_epsilon_inv @ I_minus_AT.transpose()
        
        return magnitude_moral
    
    @classmethod
    def get_symm_skeleton(cls, skeleton):
        """
        get symmetrized skeleton, assuming it is originally lower/upper diagonal
        """
        cls.assert_triangular(skeleton)
        skeleton_sym = skeleton + skeleton.transpose()
        return skeleton_sym
    
    @classmethod
    def get_support_set(cls, skeleton):
        return set(zip(*np.nonzero(skeleton)))
    
    @classmethod
    def get_moral_delta(cls, skeleton):
        """
        get the delta between the moralized graph and the "symmetrized" skeleton
        """
        cls.assert_triangular(skeleton)
        skeleton_sym = skeleton + skeleton.transpose()
        skeleton_moral = cls.get_moral_graph(skeleton)
        
        support_moral = cls.get_support_set(skeleton_moral)
        support_skeleton_sym = cls.get_support_set(skeleton_sym)
        
        assert support_skeleton_sym - support_moral == set()
        delta = support_moral - support_skeleton_sym
        
        return delta
    
    @classmethod
    def draw_graph(cls, G, is_directed=True, filepath = None):
        
        if isinstance(G, np.ndarray):
            G = nx.from_numpy_array(G.transpose(), create_using=nx.DiGraph if is_directed else nx.Graph)
        
        n_nodes = len(G.nodes())
        avg_degree = sum([v for _, v in G.degree()])/n_nodes
        
        plt.rcParams["figure.figsize"] = (10,10)

        if not nx.is_tree(G):
            pos = nx.circular_layout(G)
        else:
            pos = graphviz_layout(G, prog="dot")
        nx.draw_networkx_nodes(G, pos, node_color="tab:blue", edgecolors="tab:gray",alpha=0.3)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color="tab:red", alpha=0.5, arrows = True)
    
        plt.axis("off")
        plt.title(f'n_nodes = {n_nodes}, avg_degree = {avg_degree:.2f}')
        
        if filepath is not None:
            plt.savefig(filepath,facecolor='w')
            plt.close()
        else:
            plt.show()
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        

class LinearSEMBase(SkeletonSimulator):
    """
    simulate according to a Linear DAG
    for each node, the conditional mean is a linear function of its parents
    """
    def __init__(self, params):
        super(LinearSEMBase, self).__init__(params['n_nodes'])
    
    @classmethod
    def convert_Omega_to_graph(cls, Omega):
        """
        infer the betas from Omega; e.g., eqn (4) in https://stat.ethz.ch/Manuscripts/buhlmann/gelato.pdf
        """
        OmegaDiagInv = np.diag(1.0/np.diag(Omega))
        graph = (-1) * OmegaDiagInv @ Omega
        np.fill_diagonal(graph, 0)
        return graph
    
    def generate_DAG_coefs(self, sig_low, sig_high, skeleton=None):
        """
        generate the coefs of the DAGs
        """
        if skeleton is None:
            skeleton = np.ones((self.n_nodes, self.n_nodes))
        A = np.random.choice([-1,1], size=(self.n_nodes, self.n_nodes)) * np.random.uniform(low=sig_low, high=sig_high, size=(self.n_nodes, self.n_nodes))
        return A * skeleton

    def simulate_samples(self, n_samples, A, noise_sd=1.0):
        """
        population model is given by x = Ax + epsilon
        sample is given by X = XA^T + E; i.e., X = E[(I-A^T)^{-1}]
        """
        noise = noise_sd * np.random.normal(size=(n_samples, self.n_nodes))
        ## note that (I-AT)^{-1} == [(I-A)^{-1}]^T
        I_minus_AT_inv = np.linalg.inv(np.identity(self.n_nodes) - A.transpose())
        x_samples = np.matmul(noise, I_minus_AT_inv)
        
        return x_samples
    
    def simulate(self, n_samples):
        
        skeleton = self.generate_DAG_skeleton(graph_type=self.params['graph_type'],
                            n_children_low=self.params['n_children_low'],
                            n_children_high=self.params['n_children_high'])
        A = self.generate_DAG_coefs(self.params['sig_low'], self.params['sig_high'], skeleton=G_00)
        
        x = self.simulate_samples(n_samples, A, noise_sd=self.params['noise_sd'])
        
        Omega = self.get_moral_graph_magnitude(A, Omega_epsilon=self.params['noise_sd']*np.identity(self.n_nodes))
        graph = self.convert_Omega_to_graph(Omega)
        symSkel = self.get_symm_skeleton(skeleton)
            
        return {'graph': graph, 'Omega': Omega, 'A': A, 'x': x, 'symSkel': symSkel}
        
    
class LinearSEM2Mix(LinearSEMBase):
    """
    simulator for the following setting:
    (0) base skeleton. G^0 and G^1 both satisfy some tree structure
    (1) for each sample (x_i, z_i): sample z_i from unif(0,1);
        - for z_i < 0.25 or z_i > 0.75, take G^0 and G^1, respectively
        - for z_i in [0.25, 0.75], mixture
    """
    def __init__(self, params):
        super(LinearSEM2Mix, self).__init__(params)
        self.params = params
        
    def simulate_base_As(self):
        G_00 = self.generate_DAG_skeleton(graph_type=self.params['graph_type'],
                                    n_children_low=self.params['n_children_low'],
                                    n_children_high=self.params['n_children_high'])
        G_01 = self.generate_DAG_skeleton(graph_type=self.params['graph_type'],
                                    n_children_low=self.params['n_children_low'],
                                    n_children_high=self.params['n_children_high'])
        
        A_00 = self.generate_DAG_coefs(self.params['sig_low'], self.params['sig_high'], skeleton=G_00)
        A_01 = self.generate_DAG_coefs(self.params['sig_low'], self.params['sig_high'], skeleton=G_01)
        return A_00, A_01
    
    def simulate_one_sample(self, A_00, A_01):
        
        z = np.random.uniform(low=-1.0,high=1.0,size=(2,))
        
        if 0 < z[0] < 0.5:
            A, regime_id = A_00, 1
        elif -0.5 < z[0] <= 0.0:
            A, regime_id = A_01, 2
        else:
            mixing_pct = min(max(0.25,z[1]**2),0.75)
            A = mixing_pct * A_00 + (1-mixing_pct) * A_01
            regime_id = 3
            
        x = self.simulate_samples(n_samples=1, A=A, noise_sd=self.params['noise_sd'])
        return x, z, A, regime_id
        
    def simulate(self, n_samples):
        
        A_00, A_01 = self.simulate_base_As()
        x_lst, z_lst, regime_lst = [], [], []
        A_lst, Omega_lst, graph_lst, symSkel_lst = [], [], [], []
        
        for _ in range(n_samples):
            x, z, A, regime_id = self.simulate_one_sample(A_00, A_01)
            x_lst.append(x)
            z_lst.append(z)
            regime_lst.append(regime_id)
            A_lst.append(A)
            Omega = self.get_moral_graph_magnitude(A, Omega_epsilon=self.params['noise_sd']*np.identity(self.n_nodes))
            Omega_lst.append(Omega)
            graph_lst.append(self.convert_Omega_to_graph(Omega))
            symSkel_lst.append(self.get_symm_skeleton(1*(A!=0)))
            
        A = np.stack(A_lst, axis=0)
        Omega = np.stack(Omega_lst, axis=0)
        graph = np.stack(graph_lst, axis=0)
        symSkel = np.stack(symSkel_lst, axis=0)

        x = np.concatenate(x_lst, axis=0)
        z = np.stack(z_lst, axis=0)
        regime = np.array(regime_lst)
        
        return {'graph': graph, 'Omega': Omega, 'A': A, 'x': x, 'z': z, 'symSkel': symSkel, 'regime': regime}
        

class HermiteSEMBase(SkeletonSimulator):
    """
    simulate according to a DAG where the nodewise regression is given in the form of Hermite polynomial of its parents
    for each node, the conditional mean is given as follows
    x_j = \sum_k \sum_{i=1}^n b_{jk,i} * psi_i(x_k) + noise where psi_i() is an i-th order Hermite functions
    """
    def __init__(self, params):
        super(HermiteSEMBase, self).__init__(params['n_nodes'])
        self.params = params
        if params['hermite_type'] == 'fn':
            self.hermite_eval_fn = hermite_functions
            self.hermite_linear_extractor = hermite_function_linear_coefs
        elif params['hermite_type'] == 'poly':
            self.hermite_eval_fn = hermite_polynomials
            self.hermite_linear_extractor = hermite_function_linear_coefs
        else:
            raise ValueError(f'unrecognized `hermite_type` {params["hermite_type"]}; choose between `fn` and `poly`')
        
        
    def generate_hermite_coefs(self, n_hermite, sig_ranges, skeleton=None):
        """
        generate the "extra" coefs of the Hermite polynomial/functions
        Return:
        - np.array of size [n_nodes, n_nodes, n_hermite+1] with [j,k,:] corresponding to coef of the hermite fns psi_n(x[k]) or H_n(x[k]) in the case of hermite polynomial, when j is the response
        """
        if skeleton is None:
            skeleton = np.ones((self.n_nodes, self.n_nodes))
        assert len(sig_ranges) == n_hermite + 1
        
        coefs = []
        for hermite_id in range(n_hermite + 1):
            sig_low, sig_high = sig_ranges[hermite_id]
            if sig_low == sig_high:
                coef_curr = sig_low * np.ones((self.n_nodes, self.n_nodes))
            elif sig_low < sig_high:
                coef_curr = np.random.choice([-1,1],size=(self.n_nodes, self.n_nodes)) * np.random.uniform(low=sig_low,high=sig_high,size=(self.n_nodes, self.n_nodes))
            else:
                raise ValueError(f'hermite_id={hermite_id}: sig_low > sig_high; invalid')
            coef_curr *= skeleton ## superimpose so that in the case the j,k is not part of the skeleton, it's always set at zero
            coefs.append(coef_curr)
        return np.stack(coefs, axis=-1)
       
       
    def extract_linear_approx(self, coefs_for_hermite, include_0th_hermite=False):
        """
        Argv:
        - coefs_for_hermite: [n_nodes, n_nodes, n_hermite + 1] that stores the coef of the hermite fns/polynomials
        Return:
        - linear approximation: [n_nodes, n_nodes], so that x \approx Ax + noise;
        """
        n_hermite = coefs_for_hermite.shape[-1] - 1
        n_nodes = coefs_for_hermite.shape[0]
        offset = 1 if not include_0th_hermite else 0
        
        linear_coef_vec = self.hermite_linear_extractor(n_hermite, all_n=True) # (n_hermite+1, )
        coef_mtx = np.empty((n_nodes, n_nodes))
        for j in range(n_nodes):
            for k in range(n_nodes):
                coef_mtx[j,k] = np.dot(coefs_for_hermite[j,k,offset:], linear_coef_vec[offset:])
        return coef_mtx
        
        
    def simulate_one_sample(self, coefs_for_hermite, noise_sd=1, include_0th_hermite=False):
        """
        generate one sample according to the DAG specification
        """
        n_hermite = coefs_for_hermite.shape[-1] - 1
        offset = 1 if not include_0th_hermite else 0
        
        noise = noise_sd * np.random.normal(size=(self.n_nodes,))
        x = np.zeros_like(noise)
        for j in range(self.n_nodes):
            signal = 0
            for k in range(j):
                coef = coefs_for_hermite[j,k,offset:]
                if np.sum(np.abs(coef)) == 0:
                    continue
                basis = self.hermite_eval_fn(n_hermite, x[k])[offset:]
                val = np.dot(basis, coef)
                if DEBUG:
                    print(f'child_id={j}; parent_id={k}; parent_val={x[k]:.2f}, f(parent_val)={val:.2f}')
                signal += val
            x[j] = signal + noise[j]
            ## validate the generated node
            assert not ((np.isnan(x[j]) or np.isinf(x[j])))
            if DEBUG:
                print(f'finished node_id={j}; node_val={x[j]:.2f}')
                assert np.abs(x[j]) < 100, f'node_id={j}; node_val={x[j]:.2f}; exploded'
                
        return x


    def simulate_samples(self, n_samples, coefs_for_hermite, noise_sd=1, verbose=10000, include_0th_hermite=False):
        
        t0 = time.time()
        x_samples = np.empty((n_samples, self.n_nodes))
        for sample_id in range(n_samples):
            x_samples[sample_id] = self.simulate_one_sample(coefs_for_hermite, noise_sd, include_0th_hermite=include_0th_hermite)
            if (sample_id+1) % verbose == 0:
                print(f'done simulating {sample_id+1}/{n_samples} samples; time elapsed={time.time()-t0:.1f}s')
                t0 = time.time()
        return x_samples


    def simulate(self, n_samples, verbose=10000):
        
        skeleton = self.generate_DAG_skeleton(graph_type=self.params['graph_type'],
                            n_children_low=self.params['n_children_low'],
                            n_children_high=self.params['n_children_high'])
        graph = self.get_moral_graph(skeleton)
        hermite_coefs = self.generate_hermite_coefs(self.params['n_hermite'], self.params['sig_ranges'], skeleton=skeleton)
        
        include_0th_hermite = self.params.get('include_0th_hermite', False)
        
        x = self.simulate_samples(n_samples, hermite_coefs, noise_sd=self.params['noise_sd'], verbose=verbose, include_0th_hermite=include_0th_hermite)
        ## get also linear approximation and the corresponding moralized graph
        A = self.extract_linear_approx(hermite_coefs, include_0th_hermite=include_0th_hermite)
        Omega = self.get_moral_graph_magnitude(A, Omega_epsilon=self.params['noise_sd']*np.identity(self.n_nodes))
        symSkel = self.get_symm_skeleton(skeleton)
        
        return {'x': x, 'graph': graph, 'A': A, 'Omega': Omega, 'symSkel': symSkel}


class HermiteSEM2Mix(HermiteSEMBase):
    """
    simulator for the following setting:
    (0) base skeleton. G^0 and G^1 both satisfy some tree structure
    (1) for each sample (x_i, z_i): sample z_i (scaler) from unif(0,1);
        - for z_i < 0.25 or z_i > 0.75, take G^0 and G^1, respectively
        - for z_i in [0.25, 0.75], mixture
    """
    def __init__(self, params):
        super(HermiteSEM2Mix, self).__init__(params)
        self.params = params
        
    def simulate_base_coefs(self):
        G_00 = self.generate_DAG_skeleton(graph_type='tree',
                                    n_children_low=self.params['n_children_low'],
                                    n_children_high=self.params['n_children_high'])
        G_01 = self.generate_DAG_skeleton(graph_type='tree',
                                    n_children_low=self.params['n_children_low'],
                                    n_children_high=self.params['n_children_high'])
        hermite_coefs_00 = self.generate_hermite_coefs(self.params['n_hermite'], self.params['sig_ranges'], skeleton=G_00)
        hermite_coefs_01 = self.generate_hermite_coefs(self.params['n_hermite'], self.params['sig_ranges'], skeleton=G_01)
        return hermite_coefs_00, hermite_coefs_01
    
    def simulate_single_sample(self, hermite_coefs_00, hermite_coefs_01):
        """
        note: in the base class, there is a method >simulate_one_sample()
        """
        z = np.random.uniform(low=-1.0,high=1.0,size=(2,))
        if 0 < z[0] < 0.5:
            hermite_coefs = hermite_coefs_00
            regime_id = 1
        elif -0.5 < z[0] <= 0.0:
            hermite_coefs = hermite_coefs_01
            regime_id = 2
        else:
            mixing_pct = min(max(z[1]**2,0.25),0.75)
            hermite_coefs = mixing_pct * hermite_coefs_00 + (1-mixing_pct) * hermite_coefs_01
            regime_id = 3
            
        x = self.simulate_one_sample(hermite_coefs, self.params['noise_sd'], include_0th_hermite=False)
        return x, z, hermite_coefs, regime_id
        
    def simulate(self, n_samples, verbose=10000):
        
        hermite_coefs_00, hermite_coefs_01 = self.simulate_base_coefs()
        x_lst, z_lst, regime_lst = [], [], []
        A_lst, graph_lst, Omega_lst, symSkel_lst = [], [], [], []
        # hermite_coef_lst = []
        
        t0 = time.time()
        for sample_id in range(n_samples):
            x, z, hermite_coef, regime_id = self.simulate_single_sample(hermite_coefs_00, hermite_coefs_01)
            
            x_lst.append(x)
            z_lst.append(z)
            regime_lst.append(regime_id)
            
            A = self.extract_linear_approx(hermite_coef, include_0th_hermite=False)
            A_lst.append(A)
            
            Omega = self.get_moral_graph_magnitude(A, Omega_epsilon=self.params['noise_sd']*np.identity(self.n_nodes))
            Omega_lst.append(Omega)
            
            skeleton = 1 * (np.abs(hermite_coef[:,:,-1]) != 0)
            graph_lst.append(self.get_moral_graph(skeleton))
            symSkel_lst.append(self.get_symm_skeleton(skeleton))
            
            # hermite_coef_lst.append(hermite_coef)
            if (sample_id+1) % verbose == 0:
                print(f'done simulating {sample_id+1}/{n_samples} samples; time elapsed={time.time()-t0:.1f}s')
                t0 = time.time()
            
        A = np.stack(A_lst, axis=0)
        Omega = np.stack(Omega_lst, axis=0)
        graph = np.stack(graph_lst, axis=0)
        symSkel = np.stack(symSkel_lst, axis=0)

        x = np.stack(x_lst, axis=0)
        z = np.stack(z_lst, axis=0)
        regime = np.array(regime_lst)
        #hermite_coef = np.stack(hermite_coef_lst, axis=0)
        return {'graph': graph, 'Omega': Omega, 'A': A, 'x': x, 'z': z, 'symSkel': symSkel, 'regime': regime} #, 'hermite_coef': hermite_coef}
