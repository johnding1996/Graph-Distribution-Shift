import argparse
# import utils_parsing as parse
import os

import random
import copy
import json

import torch
import numpy as np

from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from .utils import prepare_dataset, get_custom_edge_list
from .utils_data_prep import separate_data, separate_data_given_split
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from .utils_ids import subgraph_counts2ids
from .utils_graph_processing import subgraph_isomorphism_edge_counts, subgraph_isomorphism_vertex_counts, \
    induced_edge_automorphism_orbits, edge_automorphism_orbits, automorphism_orbits
from .utils_encoding import encode


class GSN():

    def __init__(self, dataset_name, dataset_group, induced, id_type, k):

        super(GSN, self).__init__()

        # set seeds to ensure reproducibility
        self.seed = 0
        self.split_seed = 0
        self.np_seed = 0

        # set multiprocessing to true in order to do the precomputation in parallel
        self.multiprocessing = False
        self.num_processes = 64

        ###### data loader parameters
        self.num_workers = 0
        self.num_threads = 1
        ###### these are to select the dataset:
        # - dataset can be bionformatics or social and states the class;
        # - name is for the specific problem itself
        self.dataset = dataset_group
        self.dataset_name = dataset_name

        ######  set degree_as_tag to True to use the degree as node features;
        # set retain_features to True to keep the existing features as well;
        self.degree_as_tag = False
        self.retain_features = False

        ###### used only for ogb to reproduce the different configurations,
        # i.e. additional features (full) or not (simple)
        self.features_scope = 'full'

        # denotes the aggregation used by the virtual node
        # parser.add_argument('--vn_pooling', type=str, default='sum')
        # parser.add_argument('--input_vn_encoder', type=str, default='one_hot_encoder')
        # parser.add_argument('--d_out_vn_encoder', type=int, default=None)
        # parser.add_argument('--d_out_vn', type=int, default=None)
        ###### substructure-related parameters:
        # - id_type: substructure family
        # - induced: graphlets vs motifs
        # - edge_automorphism: induced edge automorphism or line graph edge automorphism (slightly larger group than the induced edge automorphism)
        # - k: size of substructures that are used; e.g. k=3 means three nodes
        # - id_scope: local vs global --> GSN-e vs GSN-v

        self.id_type = id_type
        self.induced = induced
        self.edge_automorphism = 'induced'
        self.k = k
        self.id_scope = 'local'
        self.custom_edge_list = None
        self.directed = False
        self.directed_orbits = False

        ###### encoding args: different ways to encode discrete data
        self.id_encoding = 'one_hot_unique'
        self.degree_encoding = 'one_hot_unique'

        # binning and minmax encoding parameters. NB: not used in our experimental evaluation
        self.id_bins = None
        self.degree_bins = None
        self.id_strategy = 'uniform'
        self.degree_strategy = 'uniform'
        self.id_range = None
        self.degree_range = None
        # parser.add_argument('--id_embedding', type=str, default='one_hot_encoder')
        # parser.add_argument('--d_out_id_embedding', type=int, default=None)
        # parser.add_argument('--degree_embedding', type=str, default='one_hot_encoder')
        # parser.add_argument('--d_out_degree_embedding', type=int, default=None)

        # parser.add_argument('--input_node_encoder', type=str, default='None')
        # parser.add_argument('--d_out_node_encoder', type=int, default=None)
        # parser.add_argument('--edge_encoder', type=str, default='None')
        # parser.add_argument('--d_out_edge_encoder', type=int, default=None)
        ###### model to be used and architecture parameters, in particular
        # - d_h: is the dimension for internal mlps, set to None to
        #   make it equal to d_out
        # - final_projection: is for jumping knowledge, specifying
        #   which layer is accounted for in the last model stage, if
        #   the list has only one element, that that value gets applied
        #   to all the layers
        # - jk_mlp: set it to True to use an MLP after each jk layer, otherwise a linear layer will be used
        # parser.add_argument('--model_name', type=str, default='GSN_sparse')
        # parser.add_argument('--random_features',  type=parse.str2bool, default=False)
        # parser.add_argument('--num_mlp_layers', type=int, default=2)
        # parser.add_argument('--d_h', type=int, default=None)
        # parser.add_argument('--activation_mlp', type=str, default='relu')
        # parser.add_argument('--bn_mlp', type=parse.str2bool, default=True)
        # parser.add_argument('--num_layers', type=int, default=2)
        self.num_layers = 2
        # parser.add_argument('--d_msg', type=int, default=None)
        # parser.add_argument('--d_out', type=int, default=16)
        # parser.add_argument('--bn', type=parse.str2bool, default=True)
        # parser.add_argument('--dropout_features', type=float, default=0)
        # parser.add_argument('--activation', type=str, default='relu')
        # parser.add_argument('--train_eps', type=parse.str2bool, default=False)
        # parser.add_argument('--aggr', type=str, default='add')
        # parser.add_argument('--flow', type=str, default='source_to_target')

        # parser.add_argument('--final_projection', type=parse.str2list2bool, default=[True])
        # parser.add_argument('--jk_mlp', type=parse.str2bool, default=False)
        # parser.add_argument('--residual', type=parse.str2bool, default=False)

        # parser.add_argument('--readout', type=str, default='sum')

        ###### architecture variations:
        # - msg_kind: gin (extends gin with structural identifiers), 
        #             general (general formulation with MLPs - eq 3,4 of the main paper)
        #             ogb (extends the architecture used in ogb with structural identifiers)
        # - inject*: passes the relevant variable to deeper layers akin to skip connections.
        #            If set to False, then the variable is used only as input to the first layer

        self.msg_kind = 'general'
        self.inject_ids = False
        self.inject_degrees = False
        self.inject_edge_features = True

        self.device_idx = 0

        ## ----------------------------------- argument processing
        # args, extract_ids_fn, count_fn, automorphism_fn = process_arguments(args)

        self.extract_id_fn = subgraph_counts2ids

        ###### choose the function that computes the automorphism group and the orbits #######
        if self.edge_automorphism == 'induced':
            self.automorphism_fn = induced_edge_automorphism_orbits if self.id_scope == 'local' else automorphism_orbits
        elif self.edge_automorphism == 'line_graph':
            self.automorphism_fn = edge_automorphism_orbits if self.id_scope == 'local' else automorphism_orbits
        else:
            raise NotImplementedError

        ###### choose the function that computes the subgraph isomorphisms #######
        self.count_fn = subgraph_isomorphism_edge_counts if self.id_scope == 'local' else subgraph_isomorphism_vertex_counts

        ###### choose the substructures: usually loaded from networkx,
        ###### except for 'all_simple_graphs' where they need to be precomputed,
        ###### or when a custom edge list is provided in the input by the user
        if self.id_type in ['cycle_graph',
                            'path_graph',
                            'complete_graph',
                            'binomial_tree',
                            'star_graph',
                            'nonisomorphic_trees']:
            # self.k = self.k[0]
            k_max = self.k
            k_min = 2 if self.id_type == 'star_graph' else 3
            self.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), self.id_type)

            # elif self.id_type in ['cycle_graph_chosen_k',
        #                          'path_graph_chosen_k', 
        #                          'complete_graph_chosen_k',
        #                          'binomial_tree_chosen_k',
        #                          'star_graph_chosen_k',
        #                          'nonisomorphic_trees_chosen_k']:
        #     self.custom_edge_list = get_custom_edge_list(self.k, self.id_type.replace('_chosen_k',''))
        # elif args['id_type'] in ['all_simple_graphs']:
        #     args['k'] = args['k'][0]
        #     k_max = args['k']
        #     k_min = 3
        #     filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        #     args['custom_edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)

        # elif args['id_type'] in ['all_simple_graphs_chosen_k']:
        #     filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        #     args['custom_edge_list'] = get_custom_edge_list(args['k'], filename=filename)

        # elif args['id_type'] in ['diamond_graph']:
        #     args['k'] = None
        #     graph_nx = nx.diamond_graph()
        #     args['custom_edge_list'] = [list(graph_nx.edges)]

        # elif args['id_type'] == 'custom':
        #     assert args['custom_edge_list'] is not None, "Custom edge list must be provided."

        # else:
        #     raise NotImplementedError("Identifiers {} are not currently supported.".format(args['id_type']))

        # define if degree is going to be used as a feature and when (for each layer or only at initialization)
        if self.inject_degrees:
            self.degree_as_tag = [self.degree_as_tag for _ in range(self.num_layers)]
        else:
            self.degree_as_tag = [self.degree_as_tag] + [False for _ in range(self.num_layers - 1)]

        # define if existing features are going to be retained when the degree is used as a feature
        self.retain_features = [self.retain_features] + [True for _ in range(self.num_layers - 1)]

        # replicate d_out dimensions if the rest are not defined (msg function, mlp hidden dimension, encoders, etc.)
        # and repeat hyperparams for every layer
        # if args['d_msg'] == -1:
        #     args['d_msg'] = [None for _ in range(args['num_layers'])]
        # elif args['d_msg'] is None:
        #     args['d_msg'] = [args['d_out'] for _ in range(args['num_layers'])]
        # else:
        #     args['d_msg'] = [args['d_msg'] for _ in range(args['num_layers'])]    
        # if args['d_h'] is None:
        #     args['d_h'] = [[args['d_out']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
        # else:
        #     args['d_h'] = [[args['d_h']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
        # if args['d_out_edge_encoder'] is None:
        #     args['d_out_edge_encoder'] = [args['d_out'] for _ in range(args['num_layers'])]
        # else:
        #     args['d_out_edge_encoder'] = [args['d_out_edge_encoder'] for _ in range(args['num_layers'])]
        # if args['d_out_node_encoder'] is None:
        #     args['d_out_node_encoder'] = args['d_out']
        # else:
        #     pass
        # if args['d_out_id_embedding'] is None:
        #     args['d_out_id_embedding'] = args['d_out']
        # else:
        #     pass
        # if args['d_out_degree_embedding'] is None:
        #     args['d_out_degree_embedding'] = args['d_out']
        # else:
        #     pass

        # # repeat hyperparams for every layer
        # args['d_out'] = [args['d_out'] for _ in range(args['num_layers'])]

        # args['train_eps'] = [args['train_eps'] for _ in range(args['num_layers'])]

        # if len(args['final_projection']) == 1:
        #     args['final_projection'] = [args['final_projection'][0] for _ in range(args['num_layers'])] + [True]

        # args['bn'] = [args['bn'] for _ in range(args['num_layers'])]
        # args['dropout_features'] = [args['dropout_features'] for _ in range(args['num_layers'])] + [args['dropout_features']]

        # begin here 
        evaluator = Evaluator(self.dataset_name)
        ## ----------------------------------- infrastructure

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.np_seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        print('[info] Setting all random seeds {}'.format(self.seed))

        torch.set_num_threads(self.num_threads)
        device = torch.device("cuda:" + str(self.device_idx) if torch.cuda.is_available() else "cpu")

        ## ----------------------------------- datasets: prepare and preprocess (count or load subgraph counts)

    def preprocess(self, root_path):
        subgraph_params = {'induced': self.induced,
                           'edge_list': self.custom_edge_list,
                           'directed': self.directed,
                           'directed_orbits': self.directed_orbits}
        graphs_ptg, num_classes, orbit_partition_sizes = prepare_dataset(root_path,
                                                                         self.dataset,
                                                                         self.dataset_name,
                                                                         self.id_scope,
                                                                         self.id_type,
                                                                         self.k,
                                                                         self.extract_id_fn,
                                                                         self.count_fn,
                                                                         self.automorphism_fn,
                                                                         self.multiprocessing,
                                                                         self.num_processes,
                                                                         **subgraph_params)

        ## node and edge feature dimensions
        if graphs_ptg[0].x.dim() == 1:
            num_features = 1
        else:
            num_features = graphs_ptg[0].num_features

        if hasattr(graphs_ptg[0], 'edge_features'):
            if graphs_ptg[0].edge_features.dim() == 1:
                num_edge_features = 1
            else:
                num_edge_features = graphs_ptg[0].edge_features.shape[1]
        else:
            num_edge_features = None

        if self.dataset == 'chemical' and self.dataset_name == 'ZINC':
            d_in_node_encoder, d_in_edge_encoder = torch.load(
                os.path.join(root_path, 'processed', 'num_feature_types.pt'))
            d_in_node_encoder, d_in_edge_encoder = [d_in_node_encoder], [d_in_edge_encoder]
        else:
            d_in_node_encoder = [num_features]
            d_in_edge_encoder = [num_edge_features]
        ## ----------------------------------- encode ids and degrees (and possibly edge features)

        degree_encoding = self.degree_encoding if self.degree_as_tag else None
        id_encoding = self.id_encoding if self.id_encoding != 'None' else None
        encoding_parameters = {
            'ids': {
                'bins': self.id_bins,
                'strategy': self.id_strategy,
                'range': self.id_range,
            },
            'degree': {
                'bins': self.degree_bins,
                'strategy': self.degree_strategy,
                'range': self.degree_range}}

        print("Encoding substructure counts and degree features... ", end='')
        graphs_ptg, encoder_ids, d_id, encoder_degrees, d_degree = encode(graphs_ptg,
                                                                          id_encoding,
                                                                          degree_encoding,
                                                                          **encoding_parameters)

        return graphs_ptg, encoder_ids, d_id, d_degree
