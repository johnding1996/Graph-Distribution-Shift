import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, GINConv
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

# mol
class GNN(torch.nn.Module):
    """
    Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    Input:
        - batched Pytorch Geometric graph object
    Output:
        - prediction (Tensor): float torch tensor of shape (num_graphs, num_tasks)
    """

    def __init__(self, gnn_type='gin', dataset_group='mol', num_tasks=128, num_layers = 5, emb_dim = 300, dropout = 0.5):
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

        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        if num_tasks is None:
            self.d_out = self.emb_dim
        else:
            self.d_out = self.num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if self.gnn_type == 'gin_virtual':
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, dataset_group=self.dataset_group, gnn_type='gin', drop_ratio=dropout)
        elif self.gnn_type == 'gcn_virtual' :
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, dataset_group=self.dataset_group, gnn_type='gcn', drop_ratio=dropout)
        elif self.gnn_type == 'gin' :
            self.gnn_node = GNN_node(num_layers, emb_dim, dataset_group=self.dataset_group, gnn_type='gin', drop_ratio=dropout)
        elif self.gnn_type == 'gcn' :
            self.gnn_node = GNN_node(num_layers, emb_dim, dataset_group=self.dataset_group, gnn_type='gcn', drop_ratio=dropout)
        else :
            raise NotImplementedError

        # Pooling function to generate whole-graph embeddings
        self.pool = global_mean_pool
        if num_tasks is None:
            self.graph_pred_linear = None
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        if self.graph_pred_linear is None:
            return h_graph
        else:
            return self.graph_pred_linear(h_graph)


### GIN convolution along the graph structure
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


### GCN convolution along the graph structure
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


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, dataset_group='mol', gnn_type = 'gin', drop_ratio = 0.5, JK = "last", residual = False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()

        self.dataset_group = dataset_group
        self.gnn_type = gnn_type

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.dataset_group == 'mol' :
            self.node_encoder = AtomEncoder(emb_dim)
        elif self.dataset_group == 'ppa' :
            self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding
        elif self.dataset_group == 'RotatedMNIST' :
            self.node_encoder = torch.nn.Linear(3, emb_dim)
        else :
            raise NotImplementedError


        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                if self.dataset_group == 'RotatedMNIST' :
                    mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                                   torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
                                                   torch.nn.Linear(2 * emb_dim, emb_dim))
                    self.convs.append(GINConv(mlp, train_eps=True))
                else :
                    self.convs.append(GINConvNew(emb_dim, self.dataset_group))
            elif gnn_type == 'gcn':
                if self.dataset_group == 'RotatedMNIST' :
                    self.convs.append(GCNConv(emb_dim, emb_dim))
                else :
                    self.convs.append(GCNConvNew(emb_dim, self.dataset_group))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

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
    def __init__(self, num_layer, emb_dim, dataset_group='mol', gnn_type = 'gin', drop_ratio = 0.5, JK = "last", residual = False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()

        self.dataset_group = dataset_group
        self.gnn_type = gnn_type

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.dataset_group == 'mol' :
            self.node_encoder = AtomEncoder(emb_dim)
        elif self.dataset_group == 'ppa' :
            self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding
        elif self.dataset_group == 'RotatedMNIST' :
            self.node_encoder = torch.nn.Linear(3, emb_dim)
        else :
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
                if self.dataset_group == 'RotatedMNIST' :
                    mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                                   torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
                                                   torch.nn.Linear(2 * emb_dim, emb_dim))
                    self.convs.append(GINConv(mlp, train_eps=True))
                else :
                    self.convs.append(GINConvNew(emb_dim, self.dataset_group))
            elif gnn_type == 'gcn':
                if self.dataset_group == 'RotatedMNIST' :
                    self.convs.append(GCNConv(emb_dim, emb_dim))
                else :
                    self.convs.append(GCNConvNew(emb_dim, self.dataset_group))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                    torch.nn.ReLU(), \
                                    torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                    torch.nn.ReLU()))

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
