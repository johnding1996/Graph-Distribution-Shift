import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import degree


from typing import Optional
from torch_geometric.typing import OptTensor
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import remove_self_loops, add_self_loops, segregate_self_loops
from torch_sparse import coalesce

import pdb


class GINConvNew(MessagePassing):
    def __init__(self, emb_dim, dataset_group):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConvNew, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if dataset_group == 'mol' :
            self.edge_encoder = BondEncoder(emb_dim = emb_dim)
        else :
            self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)

        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConvNew(MessagePassing):
    def __init__(self, emb_dim, dataset_group):
        super(GCNConvNew, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        if dataset_group == 'mol' :
            self.edge_encoder = BondEncoder(emb_dim = emb_dim)
        else :
            self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class ChebConvNew(MessagePassing):
    def __init__(self, emb_dim: int, K: int, dataset_group: str,
                 normalization: Optional[str] = 'sym', bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConvNew, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if dataset_group == 'mol' :
            self.edge_encoder = BondEncoder(emb_dim = emb_dim)
        else :
            self.edge_encoder = torch.nn.Linear(7, emb_dim)

        self.emb_dim = emb_dim
        self.normalization = normalization
        self.lins = torch.nn.ModuleList([
            torch.nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(emb_dim))
        else:
            self.register_parameter('bias', None)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_attr: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        edge_embedding = self.edge_encoder(edge_attr)

        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         None, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)
        edge_index, norm = coalesce(edge_index, norm, m=x.shape[0], n=x.shape[0])
        edge_index, norm, loop_edge_index, loop_norm = segregate_self_loops(edge_index, norm)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)


        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm, size=None)+\
                   loop_norm.view(-1,1)*F.relu(x + self.root_emb.weight)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, edge_attr=edge_embedding, norm=norm, size=None)+\
                   loop_norm.view(-1,1)*F.relu(Tx_1 + self.root_emb.weight)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j+edge_attr)
