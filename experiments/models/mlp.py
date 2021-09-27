import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, gnn_type, dataset_group, num_tasks=128, num_layers=5, emb_dim=300, dropout=0.5, is_pooled=True,
                 **model_kwargs):
        """
        Args:
            - num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
            - num_layers (int): number of message passing layers of GNN
            - emb_dim (int): dimensionality of hidden channels
            - dropout (float): dropout ratio applied to hidden channels
        """
        self.dataset_group = dataset_group

        super(MLP, self).__init__()

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


        ##################################################################################
        if self.dataset_group == 'mol':
            self.node_encoder = AtomEncoder(emb_dim)
        elif self.dataset_group == 'ppa':
            self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding
        elif self.dataset_group == 'RotatedMNIST':
            self.node_encoder = torch.nn.Linear(1, emb_dim)
        elif self.dataset_group == 'ColoredMNIST' :
            self.node_encoder = torch.nn.Linear(2, emb_dim)
            self.node_encoder_cate = torch.nn.Embedding(8, emb_dim)
        elif self.dataset_group == 'SBM' :
            self.node_encoder = torch.nn.Embedding(8, emb_dim)
        elif self.dataset_group == 'UPFD':
            self.node_encoder = torch.nn.Embedding(8, emb_dim)
        else:
            raise NotImplementedError
        ###List of GNNs
        self.fcs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.fcs.append(torch.nn.Linear(emb_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        ##################################################################################

        self.pool = global_mean_pool
        # Pooling function to generate whole-graph embeddings
        if num_tasks is None:
            self.graph_pred_linear = None
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        x = batched_data.x
        if self.dataset_group == 'ColoredMNIST' :
            x = self.node_encoder(x[:, :2]) + self.node_encoder_cate(x[:, 2:].to(torch.int).squeeze())
        else :
            x = self.node_encoder(x)
        for i in range(self.num_layers) :
            x = self.fcs[i](x)
            x = self.batch_norms[i](x)
            if i == self.num_layers - 1:
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        x = self.pool(x, batched_data.batch)

        if self.graph_pred_linear is None:
            return x
        else:
            return self.graph_pred_linear(x)



