import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, GINConv, ChebConv
from ogb.graphproppred.mol_encoder import AtomEncoder

from .conv import GCNConvNew, GINConvNew, ChebConvNew

Cheb_K = 3


# mol
class GNN(torch.nn.Module):
    """
    Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    Input:
        - batched Pytorch Geometric graph object
    Output:
        - prediction (Tensor): float torch tensor of shape (num_graphs, num_tasks)
    """

    def __init__(self, gnn_type, dataset_group, num_tasks=128, num_layers=5, emb_dim=300, dropout=0.5, is_pooled=True,
                 **model_kwargs):
        """
        Args:
            - num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
            - num_layers (int): number of message passing layers of GNN
            - emb_dim (int): dimensionality of hidden channels
            - dropout (float): dropout ratio applied to hidden channels
        """
        self.gnn_type = gnn_type
        self.dataset_group = dataset_group

        super(GNN, self).__init__()

        if self.gnn_type.endswith('layers') :
            num_layers = int(self.gnn_type.split('_')[1])

        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.is_pooled = is_pooled

        if num_tasks is None:
            self.d_out = self.emb_dim
        else:
            self.d_out = self.num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.gnn_type.endswith('virtual'):
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, dataset_group=self.dataset_group,
                                                 gnn_type=self.gnn_type.split('_')[0], dropout=dropout)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim, dataset_group=self.dataset_group,
                                     gnn_type=self.gnn_type.split('_')[0], dropout=dropout)

        # Pooling function to generate whole-graph embeddings
        if self.is_pooled:
            self.pool = global_mean_pool
        else:
            self.pool = None
        if num_tasks is None:
            self.graph_pred_linear = None
        else:
            assert self.pool is not None
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)

        if self.graph_pred_linear is None:
            if self.pool is None:
                return h_node, batched_data.batch
            else:
                return self.pool(h_node, batched_data.batch)
        else:
            return self.graph_pred_linear(self.pool(h_node, batched_data.batch))


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, dataset_group='mol', gnn_type='gin', dropout=0.5, JK="last", residual=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()

        self.dataset_group = dataset_group
        self.gnn_type = gnn_type

        self.num_layer = num_layer
        self.drop_ratio = dropout
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.dataset_group == 'mol':
            self.node_encoder = AtomEncoder(emb_dim)
        elif self.dataset_group == 'ppa':
            self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding
        elif self.dataset_group == 'RotatedMNIST':
            self.node_encoder = torch.nn.Linear(1, emb_dim)
        elif self.dataset_group == 'ColoredMNIST' :
            self.node_encoder = torch.nn.Linear(2, emb_dim)
        elif self.dataset_group == 'SBM' :
            self.node_encoder = torch.nn.Embedding(8, emb_dim)
        elif self.dataset_group == 'UPFD':
            self.node_encoder = torch.nn.Linear(10, emb_dim)
        else:
            raise NotImplementedError

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                if self.dataset_group in  ['RotatedMNIST', 'ColoredMNIST', 'SBM', 'UPFD']:
                    mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                              torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
                                              torch.nn.Linear(2 * emb_dim, emb_dim))
                    self.convs.append(GINConv(mlp, train_eps=True))
                else:
                    self.convs.append(GINConvNew(emb_dim, self.dataset_group))
            elif gnn_type == 'gcn':
                if self.dataset_group in  ['RotatedMNIST', 'ColoredMNIST', 'SBM', 'UPFD']:
                    self.convs.append(GCNConv(emb_dim, emb_dim))
                else:
                    self.convs.append(GCNConvNew(emb_dim, self.dataset_group))
            elif gnn_type == 'cheb' :
                if self.dataset_group in  ['RotatedMNIST', 'ColoredMNIST', 'SBM', 'UPFD']:
                    self.convs.append(ChebConv(emb_dim, emb_dim, Cheb_K))
                else:
                    self.convs.append(ChebConvNew(emb_dim, Cheb_K, self.dataset_group))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def destroy_node_encoder(self):
        self.node_encoder = None

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # FLAG injects perturbation
        if self.node_encoder is None:
            h_list = [x]
        else:
            h_list = [self.node_encoder(x) + perturb if perturb is not None else self.node_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, dataset_group='mol', gnn_type='gin', dropout=0.5, JK="last", residual=False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()

        self.dataset_group = dataset_group
        self.gnn_type = gnn_type

        self.num_layer = num_layer
        self.drop_ratio = dropout
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.dataset_group == 'mol':
            self.node_encoder = AtomEncoder(emb_dim)
        elif self.dataset_group == 'ppa':
            self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding
        elif self.dataset_group == 'RotatedMNIST':
            self.node_encoder = torch.nn.Linear(1, emb_dim)
        elif self.dataset_group == 'ColoredMNIST' :
            self.node_encoder = torch.nn.Linear(2, emb_dim)
        elif self.dataset_group == 'SBM' :
            self.node_encoder = torch.nn.Embedding(8, emb_dim)
        elif self.dataset_group == 'UPFD':
            self.node_encoder = torch.nn.Linear(10, emb_dim)
        else:
            raise NotImplementedError

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                if self.dataset_group in  ['RotatedMNIST', 'ColoredMNIST', 'SBM', 'UPFD']:
                    mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                              torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
                                              torch.nn.Linear(2 * emb_dim, emb_dim))
                    self.convs.append(GINConv(mlp, train_eps=True))
                else:
                    self.convs.append(GINConvNew(emb_dim, self.dataset_group))
            elif gnn_type == 'gcn':
                if self.dataset_group in  ['RotatedMNIST', 'ColoredMNIST', 'SBM', 'UPFD']:
                    self.convs.append(GCNConv(emb_dim, emb_dim))
                else:
                    self.convs.append(GCNConvNew(emb_dim, self.dataset_group))
            elif gnn_type == 'cheb' :
                if self.dataset_group in  ['RotatedMNIST', 'ColoredMNIST', 'SBM', 'UPFD']:
                    self.convs.append(ChebConv(emb_dim, emb_dim, Cheb_K))
                else:
                    self.convs.append(ChebConvNew(emb_dim, Cheb_K, self.dataset_group))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                    torch.nn.ReLU(), \
                                    torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                    torch.nn.ReLU()))

    def destroy_node_encoder(self):
        self.node_encoder = None

    def forward(self, batched_data, perturb=None):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # FLAG injects perturbation
        if self.node_encoder is None:
            h_list = [x]
        else:
            h_list = [self.node_encoder(x) + perturb if perturb is not None else self.node_encoder(x)]

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.drop_ratio, training=self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
