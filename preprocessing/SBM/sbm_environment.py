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


def block_model_adj(com, p_r, q):
    n=len(com)
    W=np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i+1,n):
            if com[i] == com[j]:
                prob = p_r[com[i]]
            else:
                prob = q
            if np.random.binomial(1, prob) == 1:
                W[i,j] = 1
                W[j,i] = 1     
    return W


def block_model(sel_r, clust_size_min, clust_size_max, p_r, q):  
    com = []
    for r in sel_r:
        clust_size = np.random.randint(clust_size_min, clust_size_max, size=1)[0]
        com.append(np.repeat(r, clust_size, axis=0))
    com = np.concatenate(com)
    W = block_model_adj(com, p_r, q)
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
    
    
SMB_ENVIRONMENT_CONFIG = {
    'r1': 4,
    'r2': 3,
    's1': 3,
    's2': 2,
    'p_r': np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
    'size_min': 20,
    'size_max': 40,
    'q': 0.1,
    'ntypes': 8,
}


DOMAIN_SELECTION_DICT = dict(zip([frozenset(sel_r1) for sel_r1 in combinations(range(SMB_ENVIRONMENT_CONFIG['r1']), SMB_ENVIRONMENT_CONFIG['s1'])], range(comb(SMB_ENVIRONMENT_CONFIG['r1'], SMB_ENVIRONMENT_CONFIG['s1'], exact=True))))
CLASS_SELECTION_DICT = dict(zip([frozenset(sel_r2) for sel_r2 in combinations(range(SMB_ENVIRONMENT_CONFIG['r2']), SMB_ENVIRONMENT_CONFIG['s2'])], range(comb(SMB_ENVIRONMENT_CONFIG['r2'], SMB_ENVIRONMENT_CONFIG['s2'], exact=True))))


def encode_domain_selection(sel_r1):
    return DOMAIN_SELECTION_DICT[frozenset(sel_r1)]


def encode_class_selection(sel_r2):
    return CLASS_SELECTION_DICT[frozenset(sel_r2)]


class SBMEnvironmentGraph():
    def __init__(self, config=SMB_ENVIRONMENT_CONFIG):
        # parameters
        self.r1  = SMB_ENVIRONMENT_CONFIG['r1']
        self.r2  = SMB_ENVIRONMENT_CONFIG['r2']
        self.s1  = SMB_ENVIRONMENT_CONFIG['s1']
        self.s2  = SMB_ENVIRONMENT_CONFIG['s2']
        assert self.s1 > 0 and self.s1 < self.r1
        assert self.s2 > 0 and self.s2 < self.r2
        self.p_r = SMB_ENVIRONMENT_CONFIG['p_r']
        self.p_r1 = self.p_r[0::2]
        self.p_r2 = self.p_r[1::2]
        assert len(self.p_r1) == self.r1
        assert len(self.p_r2) == self.r2
        self.size_min = SMB_ENVIRONMENT_CONFIG['size_min']
        self.size_max = SMB_ENVIRONMENT_CONFIG['size_max']
        self.q = SMB_ENVIRONMENT_CONFIG['q']
        self.ntypes = SMB_ENVIRONMENT_CONFIG['ntypes']
        # data
        self.x = None
        self.y = None
        self.edge_index = None
        self.domain_id = None
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
        assert self.domain_id.ndim == 0
        
        assert self.x.dtype == torch.int64
        assert self.y.dtype == torch.int64
        assert self.edge_index.dtype == torch.int64
        assert self.domain_id.dtype == np.int64        
        
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
        # select environment (domain defining) communities
        sel_r1 = np.random.choice(self.r1, self.s1, replace=False)
        domain_id = encode_domain_selection(sel_r1)
        
        # select the target (class defining) communities
        sel_r2 = np.random.choice(self.r2, self.s2, replace=False)
        y = encode_class_selection(sel_r2)
        
        # combine the sel and edge prob lists
        sel_r2 = sel_r2 + self.r1
        sel_r = np.concatenate([sel_r1, sel_r2])
        p_r = np.concatenate([self.p_r1, self.p_r2])
        
        # generate adj and com list
        W, com = block_model(sel_r, self.size_min, self.size_max, self.p_r, self.q)
        
        # generate iid random node features 
        x = np.random.randint(self.ntypes, size=W.shape[0])
        
        # shuffle
        W, x = schuffle(W, x)
        
        # conversion
        self.x = torch.from_numpy(x).long()
        self.y = torch.tensor(y).long()
        self.edge_index = torch.from_numpy(adj_to_edge_list(W)).long()
        self.domain_id = np.array(domain_id, dtype=np.int64)

        
def generate_func(i):
    return SBMEnvironmentGraph()
    
        
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