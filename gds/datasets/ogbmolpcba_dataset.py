import os

import numpy as np
import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data.dataloader import Collater as PyGCollater

from gds.datasets.gds_dataset import GDSDataset
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F


class OGBPCBADataset(GDSDataset):
    """
    The OGB-molpcba dataset.
    This dataset is directly adopted from Open Graph Benchmark, and originally curated by MoleculeNet.

    Supported `split_scheme`:
        - 'official' or 'scaffold', which are equivalent

    Input (x):
        Molecular graphs represented as Pytorch Geometric data objects

    Label (y):
        y represents 128-class binary labels.

    Metadata:
        - scaffold
            Each molecule is annotated with the scaffold ID that the molecule is assigned to.

    Website:
        https://ogb.stanford.edu/docs/graphprop/#ogbg-mol

    Original publication:
        @article{hu2020ogb,
            title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
            author={W. {Hu}, M. {Fey}, M. {Zitnik}, Y. {Dong}, H. {Ren}, B. {Liu}, M. {Catasta}, J. {Leskovec}},
            journal={arXiv preprint arXiv:2005.00687},
            year={2020}
        }

        @article{wu2018moleculenet,
            title={MoleculeNet: a benchmark for molecular machine learning},
            author={Z. {Wu}, B. {Ramsundar}, E. V {Feinberg}, J. {Gomes}, C. {Geniesse}, A. S {Pappu}, K. {Leswing}, V. {Pande}},
            journal={Chemical science},
            volume={9},
            number={2},
            pages={513--530},
            year={2018},
            publisher={Royal Society of Chemistry}
        }

    License:
        This dataset is distributed under the MIT license.
        https://github.com/snap-stanford/ogb/blob/master/LICENSE
    """

    _dataset_name = 'ogb-molpcba'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', random_split=False,
        subgraph=False, **dataset_kwargs):
        self._version = version
        if version is not None:
            raise ValueError('Versioning for OGB-MolPCBA is handled through the OGB package. Please set version=none.')
        # internally call ogb package
        self.ogb_dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root=root_dir)

        # set variables
        self._data_dir = self.ogb_dataset.root
        if split_scheme == 'official':
            split_scheme = 'scaffold'
        self._split_scheme = split_scheme
        self._y_type = 'float'  # although the task is binary classification, the prediction target contains nan value, thus we need float
        self._y_size = self.ogb_dataset.num_tasks
        self._n_classes = self.ogb_dataset.__num_classes__


        self._split_array = torch.zeros(len(self.ogb_dataset)).long()
        split_idx = self.ogb_dataset.get_idx_split()

        np.random.seed(0)
        dataset_size = len(self.ogb_dataset)
        if random_split:
            random_index = np.random.permutation(dataset_size)
            train_split_idx = random_index[:len(split_idx['train'])]
            val_split_idx = random_index[len(split_idx['train']):len(split_idx['train']) + len(split_idx['valid'])]
            test_split_idx = random_index[len(split_idx['train']) + len(split_idx['valid']):]
        else:
            train_split_idx = split_idx['train']
            val_split_idx = split_idx['valid']
            test_split_idx = split_idx['test']

        self._split_array[train_split_idx] = 0
        self._split_array[val_split_idx] = 1
        self._split_array[test_split_idx] = 2


        self._y_array = self.ogb_dataset.data.y
        self._metadata_fields = ['scaffold']

        metadata_file_path = os.path.join(self.ogb_dataset.root, 'raw', 'OGB-MolPCBA_group.npy')
        if not os.path.exists(metadata_file_path):
            metadata_zip_file_path = download_url(
                'https://www.dropbox.com/s/jjpr6tw34llxbpw/OGB-MolPCBA_group.zip?dl=1', self.ogb_dataset.raw_dir)
            extract_zip(metadata_zip_file_path, self.ogb_dataset.raw_dir)
            os.unlink(metadata_zip_file_path)
        self._metadata_array = torch.from_numpy(np.load(metadata_file_path)).reshape(-1, 1).long()

        if dataset_kwargs['model'] == '3wlgnn':
            self._collate = self.collate_dense
        else:
            if torch_geometric.__version__ >= '1.7.0':
                self._collate = PyGCollater(follow_batch=[], exclude_keys=[])
            else:
                self._collate = PyGCollater(follow_batch=[])

        self._metric = Evaluator('ogbg-molpcba')


        # GSN
        self.subgraph = subgraph
        if self.subgraph:
            self.id_type = dataset_kwargs['gsn_id_type']
            self.k = dataset_kwargs['gsn_k']
            from gds.datasets.gsn.gsn_data_prep import GSN
            subgraph = GSN(dataset_name='ogbg-molpcba', dataset_group='ogb', induced=True, id_type=self.id_type,
                           k=self.k)
            self.graphs_ptg, self.encoder_ids, self.d_id, self.d_degree = subgraph.preprocess(self.ogb_dataset.root)

            if self.graphs_ptg[0].x.dim() == 1:
                self.num_features = 1
            else:
                self.num_features = self.graphs_ptg[0].num_features

            if hasattr(self.graphs_ptg[0], 'edge_features'):
                if self.graphs_ptg[0].edge_features.dim() == 1:
                    self.num_edge_features = 1
                else:
                    self.num_edge_features = self.graphs_ptg[0].edge_features.shape[1]
            else:
                self.num_edge_features = None

            self.d_in_node_encoder = [self.num_features]
            self.d_in_edge_encoder = [self.num_edge_features]
        # import pdb;pdb.set_trace()
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        if self.subgraph:
            return self.graphs_ptg[int(idx)]
            
        else:
            return self.ogb_dataset[int(idx)]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (FloatTensor): Binary logits from a model
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels.
                                        Only None is supported because OGB Evaluators accept binary logits
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        assert prediction_fn is None, "OGBPCBADataset.eval() does not support prediction_fn. Only binary logits accepted"
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        results = self._metric.eval(input_dict)

        return results, f"Average precision: {results['ap']:.3f}\n"

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense(self, samples):
        def _sym_normalize_adj(adjacency):
            deg = torch.sum(adjacency, dim=0)  # .squeeze()
            deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
            deg_inv = torch.diag(deg_inv)
            return torch.mm(deg_inv, torch.mm(adjacency, deg_inv))

        # The input samples is a list of pairs (graph, label).
        node_feat_space = torch.tensor([119, 4, 12, 12, 10, 6, 6, 2, 2])
        edge_feat_space = torch.tensor([5, 6, 2])

        graph_list, y_list, metadata_list = map(list, zip(*samples))
        # multi-task y, use torch.stack instead of torch.tensor
        y, metadata = torch.stack(y_list), torch.stack(metadata_list)

        feat = []
        for graph in graph_list:
            adj = _sym_normalize_adj(to_dense_adj(graph.edge_index, max_num_nodes=graph.x.size(0)).squeeze())
            zero_adj = torch.zeros_like(adj)
            in_dim = node_feat_space.sum() + edge_feat_space.sum()

            # use node feats to prepare adj
            adj_feat = torch.stack([zero_adj for _ in range(in_dim)])
            adj_feat = torch.cat([adj.unsqueeze(0), adj_feat], dim=0)

            def convert(feature, space):
                out = []
                for i, label in enumerate(feature):
                    out.append(F.one_hot(label, space[i]))
                return torch.cat(out)

            for node, node_feat in enumerate(graph.x):
                adj_feat[1:1 + node_feat_space.sum(), node, node] = convert(node_feat, node_feat_space)
            for edge in range(graph.edge_index.shape[1]):
                target, source = graph.edge_index[0][edge], graph.edge_index[1][edge]
                edge_feat = graph.edge_attr[edge]
                adj_feat[1 + node_feat_space.sum():, target, source] = convert(edge_feat, edge_feat_space)

            feat.append(adj_feat)

        feat = torch.stack(feat)
        return feat, y, metadata
