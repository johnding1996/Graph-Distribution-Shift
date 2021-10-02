import numpy as np
import torch
import pickle
import time
from itertools import combinations
from scipy.special import comb
from multiprocessing import Pool


def schuffle(W, x):
    n = W.shape[0]
    idx=np.random.permutation(n)
    W_new=W[idx,:][:,idx]
    x_new=x[idx]
    return W_new , x_new


def block_model_adj(com, p_r, q, r_p, q_p):
    n=len(com)
    W=np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i+1,n):
            if com[i] == com[j]:
                prob = p_r[com[i]]
            else:
                if (com[i] not in r_p) and (com[j] not in r_p):
                    prob = q
                else:
                    prob = q_p
            if np.random.binomial(1, prob) == 1:
                W[i,j] = 1
                W[j,i] = 1     
    return W


def block_model(sel_r, clust_size_min, clust_size_max, p_r, q, r_p, q_p):  
    com = []
    for r in sel_r:
        clust_size = np.random.randint(clust_size_min, clust_size_max, size=1)[0]
        com.append(np.repeat(r, clust_size, axis=0))
    com = np.concatenate(com)
    W = block_model_adj(com, p_r, q, r_p, q_p)  
    return W, com


def adj_to_edge_list(W):
    n = W.shape[0]
    e = int(np.sum(W))
    edge_index = np.zeros((2, e), dtype=np.int64)
    counter = 0
    for i in range(n):
        for j in range(n):
            if W[i,j] == 1:
                edge_index[0, counter] = i
                edge_index[1, counter] = j
                counter += 1
    assert counter == e
    return edge_index
    
    
SMB_ISOLATION_CONFIG = {
    'r': 5,
    's': 3,
    'p_r': np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
    'size_min': 10,
    'size_max': 30,
    'q': 0.3,
    'q_p': 0.1,
    'ntypes': 8,
}


SELECTION_DICT = dict(zip([frozenset(sel_r) for sel_r in combinations(range(SMB_ISOLATION_CONFIG['r']), SMB_ISOLATION_CONFIG['s'])], range(comb(SMB_ISOLATION_CONFIG['r'], SMB_ISOLATION_CONFIG['s'], exact=True))))


def encode_selection(sel_r):
    return SELECTION_DICT[frozenset(sel_r)]


class SBMIsolationGraph():
    def __init__(self, config=SMB_ISOLATION_CONFIG):
        # parameters
        self.r  = SMB_ISOLATION_CONFIG['r']
        self.s  = SMB_ISOLATION_CONFIG['s']
        assert self.s > 0 and self.s < self.r
        self.p_r = SMB_ISOLATION_CONFIG['p_r']
        assert len(self.p_r) == self.r
        self.size_min = SMB_ISOLATION_CONFIG['size_min']
        self.size_max = SMB_ISOLATION_CONFIG['size_max']
        self.q = SMB_ISOLATION_CONFIG['q']
        self.q_p = SMB_ISOLATION_CONFIG['q_p']
        self.ntypes = SMB_ISOLATION_CONFIG['ntypes']
        # data
        self.x = None
        self.y = None
        self.edge_index = None
        self.domain_id = None
        self.sel_r = None
        # generate
        self.generate()
        
    def validate(self):
        assert isinstance(self.x, torch.Tensor)
        assert isinstance(self.y, torch.Tensor)
        assert isinstance(self.edge_index, torch.Tensor)
        assert isinstance(self.domain_id, np.ndarray)
        
        assert self.x.ndim == 1
        assert self.y.ndim == 0
        assert self.edge_index.ndim == 2 and self.edge_index.size(0) == 2
        assert self.domain_id.ndim == 1
        
        assert self.x.dtype == torch.int64
        assert self.y.dtype == torch.int64
        assert self.edge_index.dtype == torch.int64
        assert self.domain_id.dtype == np.int64
        
        assert set(self.x.unique().numpy()) == set(self.domain_id)
        
    def get(self):
        self.validate()
        graph_data = {
            'x': self.x,
            'y': self.y,
            'edge_index': self.edge_index,
        }
        return graph_data, self.domain_id
    
    def stat(self):
        self.validate()
        stat = {
            '# of nodes': self.x.size(0),
            '# of edges': self.edge_index.size(1)//2,
            'avg degree': self.edge_index.size(1)/self.x.size(0)/2,
        }
        return stat
    
    def generate(self):
        # select communities
        sel_r = np.random.choice(self.r, self.s, replace=False)
        sel_r = np.sort(sel_r)
        
        # isolate some communities
        y = np.random.randint(self.s + 1, size=1)[0]
        r_p = np.random.choice(sel_r, y, replace=False)
        
        # generate adj and com list
        W, com = block_model(sel_r, self.size_min, self.size_max, self.p_r, self.q, r_p, self.q_p)
        
        # conversion
        self.x = torch.from_numpy(com).long()
        self.y = torch.tensor(y).long()
        self.edge_index = torch.from_numpy(adj_to_edge_list(W)).long()
        self.domain_id = np.array(sel_r, dtype=np.int64)

        
def generate_func(i):
    return SBMIsolationGraph()
    
        
def generate_sbm_graph_list(n_samples, n_processes=1):
    np.random.seed(0)
    
    start_time = time.time()

    sbm_graph_list = []

    with Pool(processes=n_processes) as pool:
        for i, sbmg in enumerate(pool.imap_unordered(generate_func, range(n_samples))):
            sbm_graph_list.append(sbmg)
            print(f'Progress: {i/n_samples*100:.2f}%', end='\r', flush=True)
    print(' '*50, end='\r')

    print(f'Total generation time: {time.time() - start_time:.2f} secs.')
    
    return sbm_graph_list