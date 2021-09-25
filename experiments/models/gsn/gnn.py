import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_filters.GSN_edge_sparse_ogb import GSN_edge_sparse_ogb
from .graph_filters.GSN_sparse import GSN_sparse

from .graph_filters.MPNN_edge_sparse_ogb import MPNN_edge_sparse_ogb

from .models_misc import mlp, choose_activation
from .utils_graph_learning import global_add_pool_sparse, global_mean_pool_sparse, DiscreteEmbedding

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GNN_GSN(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 encoder_ids,
                 d_in_id,
                 in_edge_features=None,
                 d_in_node_encoder=None,
                 d_in_edge_encoder=None,
                 d_degree=None,
                 emb_dim=300,
                 num_layers=5,
                 dataset_group='mol'):

        super(GNN_GSN, self).__init__()

        seed = 0

        # -------------- Initializations
        self.dataset_group = dataset_group
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.model_name = 'GSN_edge_sparse_ogb'
        self.readout = 'mean'
        self.dropout_features = 0.5
        self.bn = True

        self.final_projection = [False for _ in range(self.num_layers)]
        self.final_projection.append(True)  # self.final_projection = [False, False, False, False, False, True]

        self.residual = False
        self.inject_ids = False

        id_scope = 'local'
        d_msg = emb_dim
        d_out = emb_dim
        d_h = [2 * emb_dim]
        aggr = 'add'
        flow = 'source_to_target'
        msg_kind = 'ogb'
        train_eps = False
        activation_mlp = 'relu'
        bn_mlp = True
        jk_mlp = False
        degree_embedding = None

        retain_features = [True for _ in range(self.num_layers)]
        retain_features[0] = False  # retain_features = [False, True, True, True, True]

        degree_as_tag = False
        encoders_kwargs = {'seed': seed,
                           'activation_mlp': activation_mlp,
                           'bn_mlp': bn_mlp,
                           'aggr': 'sum',
                           'features_scope': 'full'}

        if self.dataset_group == 'mol' :
            self.input_node_encoder = DiscreteEmbedding('atom_encoder',
                                                        in_features,
                                                        d_in_node_encoder,
                                                        emb_dim,
                                                        **encoders_kwargs)
            d_in = self.input_node_encoder.d_out
        elif self.dataset_group == 'ppa' :
            self.input_node_encoder = torch.nn.Embedding(1, emb_dim)
            d_in = emb_dim
        elif self.dataset_group == 'RotatedMNIST':
            self.node_encoder = torch.nn.Linear(3, emb_dim)
            d_in = emb_dim
        else :
            raise NotImplementedError

        # -------------- Edge embedding (for each GNN layer)
        self.edge_encoder = []
        d_ef = []
        if self.dataset_group == 'mol' :
            for i in range(self.num_layers):
                edge_encoder_layer = DiscreteEmbedding('bond_encoder',
                                                       in_edge_features,
                                                       d_in_edge_encoder,
                                                       emb_dim,
                                                       **encoders_kwargs)
                self.edge_encoder.append(edge_encoder_layer)
                d_ef.append(edge_encoder_layer.d_out)
        elif self.dataset_group == 'ppa' :
            for i in range(self.num_layers):
                edge_encoder_layer = torch.nn.Linear(7, emb_dim)
                self.edge_encoder.append(edge_encoder_layer)
                d_ef.append(emb_dim)
        else :
            pass

        self.edge_encoder = nn.ModuleList(self.edge_encoder)

        # -------------- Identifier embedding (for each GNN layer)
        self.id_encoder = []
        d_id = []
        num_id_encoders = self.num_layers if self.inject_ids else 1

        for i in range(num_id_encoders):
            id_encoder_layer = DiscreteEmbedding('embedding',
                                                 len(d_in_id),
                                                 d_in_id,
                                                 emb_dim,
                                                 **encoders_kwargs)
            self.id_encoder.append(id_encoder_layer)
            d_id.append(id_encoder_layer.d_out)

        self.id_encoder = nn.ModuleList(self.id_encoder)

        # -------------- Degree embedding
        self.degree_encoder = DiscreteEmbedding('None',
                                                1,
                                                1,
                                                emb_dim,
                                                **encoders_kwargs)
        d_degree = self.degree_encoder.d_out

        # -------------- GNN layers w/ bn
        self.conv = []
        self.batch_norms = []
        self.mlp_vn = []
        for i in range(self.num_layers):
            # if i > 0 and self.vn:
            #     #-------------- vn msg function     
            #     mlp_vn_temp = mlp(d_in_vn, kwargs['d_out_vn'][i-1], d_h[i], seed, activation_mlp, bn_mlp)
            #     self.mlp_vn.append(mlp_vn_temp)
            #     d_in_vn= kwargs['d_out_vn'][i-1]
            # import pdb;pdb.set_trace()
            kwargs_filter = {
                'd_in': d_in,
                'd_degree': 1,
                'degree_as_tag': degree_as_tag,
                'retain_features': retain_features[i],
                'd_msg': d_msg,
                'd_up': d_out,
                'd_h': d_h,
                'seed': seed,
                'activation_name': activation_mlp,
                'bn': bn_mlp,
                'aggr': aggr,
                'msg_kind': msg_kind,
                'eps': 0,
                'train_eps': train_eps,
                'flow': flow,
                'd_ef': d_ef[i],
                'edge_embedding': 'bond_encoder',
                'id_embedding': 'embedding',
                'extend_dims': True
            }

            use_ids = ((i > 0 and self.inject_ids) or (i == 0)) and (self.model_name == 'GSN_edge_sparse_ogb')

            if use_ids:
                # if self.dataset_group != 'mol' and self.dataset_group != 'ppa' :
                if self.dataset_group == 'mol' or self.dataset_group == 'ppa' :
                    filter_fn = GSN_edge_sparse_ogb
                else :
                    filter_fn = GSN_sparse

                kwargs_filter['d_id'] = d_id[i] if self.inject_ids else d_id[0]
                kwargs_filter['id_scope'] = id_scope
            else:
                filter_fn = MPNN_edge_sparse_ogb
            self.conv.append(filter_fn(**kwargs_filter))

            bn_layer = nn.BatchNorm1d(d_out) if self.bn else None
            self.batch_norms.append(bn_layer)

            d_in = d_out

        self.conv = nn.ModuleList(self.conv)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        # if kwargs['vn']:
        #     self.mlp_vn = nn.ModuleList(self.mlp_vn)

        # -------------- Readout
        if self.readout == 'sum':
            self.global_pool = global_add_pool_sparse
        elif self.readout == 'mean':
            self.global_pool = global_mean_pool_sparse
        else:
            raise ValueError("Invalid graph pooling type.")

        # #-------------- Virtual node aggregation operator
        # if self.vn:
        #     if kwargs['vn_pooling'] == 'sum':
        #         self.global_vn_pool = global_add_pool_sparse
        #     elif kwargs['vn_pooling'] == 'mean':
        #         self.global_vn_pool = global_mean_pool_sparse
        #     else:
        #         raise ValueError("Invalid graph virtual node pooling type.")

        if out_features is None:
            self.lin_proj = None
        else:
            self.lin_proj = nn.Linear(d_out, out_features)

        # -------------- Activation fn (same across the network)

        self.activation = choose_activation('relu')

        return

    def forward(self, data, return_intermediate=False):

        # -------------- Code adopted from https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol.
        # -------------- Modified accordingly to allow for the existence of structural identifiers

        kwargs = {}
        kwargs['degrees'] = self.degree_encoder(data.degrees)

        # -------------- edge index, initial node features enmbedding, initial vn embedding
        edge_index = data.edge_index
        # if self.vn:
        #     vn_embedding = self.vn_encoder(torch.zeros(data.batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))                                                    
        x = self.input_node_encoder(data.x)
        x_interm = [x]

        for i in range(0, len(self.conv)):

            # -------------- encode ids (different for each layer)
            kwargs['identifiers'] = self.id_encoder[i](data.identifiers) if self.inject_ids else self.id_encoder[0](
                data.identifiers)

            # -------------- edge features embedding (different for each layer)
            if hasattr(data, 'edge_features'):
                kwargs['edge_features'] = self.edge_encoder[i](data.edge_features)
            else:
                kwargs['edge_features'] = None

            # if self.vn:
            #     x_interm[i] = x_interm[i] + vn_embedding[data.batch]

            x = self.conv[i](x_interm[i], edge_index, **kwargs)

            x = self.batch_norms[i](x) if self.bn else x

            if i == len(self.conv) - 1:
                x = F.dropout(x, self.dropout_features, training=self.training)
            else:
                x = F.dropout(self.activation(x), self.dropout_features, training=self.training)

            if self.residual:
                x += x_interm[-1]

            x_interm.append(x)

            # if i < len(self.conv) - 1 and self.vn:
            #     vn_embedding_temp = self.global_vn_pool(x_interm[i], data.batch) + vn_embedding
            #     vn_embedding = self.mlp_vn[i](vn_embedding_temp)

            #     if self.residual:
            #         vn_embedding = vn_embedding + F.dropout(self.activation(vn_embedding), self.dropout_features[i], training = self.training)
            #     else:
            #         vn_embedding = F.dropout(self.activation(vn_embedding), self.dropout_features[i], training = self.training)

        prediction = 0
        for i in range(0, len(self.conv) + 1):
            if self.final_projection[i]:
                prediction += x_interm[i]

        x_global = self.global_pool(prediction, data.batch)
        if self.lin_proj == None:
            return x_global
        else:
            return self.lin_proj(x_global)
