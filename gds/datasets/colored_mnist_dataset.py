import os
import torch
import numpy as np
from gds.datasets.gds_dataset import GDSDataset
from ogb.graphproppred import Evaluator
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data.dataloader import Collater as PyGCollater
import torch_geometric
from .pyg_colored_mnist_dataset import PyGColoredMNISTDataset
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

class ColoredMNISTDataset(GDSDataset):
    _dataset_name = 'ColoredMNIST'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', random_split=False,
                 subgraph=False, **dataset_kwargs):
        self._version = version
        # internally call ogb package
        self.ogb_dataset = PyGColoredMNISTDataset(name='ColoredMNIST', root=root_dir)

        # set variables
        self._data_dir = self.ogb_dataset.root
        if split_scheme == 'official':
            split_scheme = 'color'
        self._split_scheme = split_scheme
        self._y_type = 'float'  # although the task is binary classification, the prediction target contains nan value, thus we need float
        self._y_size = self.ogb_dataset.num_tasks
        self._n_classes = self.ogb_dataset.__num_classes__

        self._split_array = torch.zeros(len(self.ogb_dataset)).long()

        self._y_array = self.ogb_dataset.data.y.unsqueeze(-1)
        self._metadata_fields = ['color', 'y']

        # https://www.dropbox.com/s/envr0eslhtssy2y/ColoredMNIST_group_expired.zip?dl=1
        metadata_file_path = os.path.join(self.ogb_dataset.raw_dir, 'ColoredMNIST_group.npy')
        if not os.path.exists(metadata_file_path):
            metadata_zip_file_path = download_url(
                'https://www.dropbox.com/s/ax1ek3yc6n1q739/ColoredMNIST_group.zip?dl=1', self.ogb_dataset.raw_dir)
            extract_zip(metadata_zip_file_path, self.ogb_dataset.raw_dir)
            os.unlink(metadata_zip_file_path)

        self._metadata_array_wo_y = torch.from_numpy(np.load(metadata_file_path)).reshape(-1, 1).long()
        self._metadata_array = torch.cat((self._metadata_array_wo_y,
                                          torch.unsqueeze(self.ogb_dataset.data.y, dim=1)), 1)

        np.random.seed(0)
        dataset_size = len(self.ogb_dataset)
        if random_split:
            random_index = np.random.permutation(dataset_size)
            train_split_idx = random_index[:int(3 / 6 * dataset_size)]
            val_split_idx = random_index[int(3 / 6 * dataset_size):int(4 / 6 * dataset_size)]
            test_split_idx = random_index[int(4 / 6 * dataset_size):]
        else:
            # use the group info split data
            train_val_group_idx, test_group_idx = range(0, 2), range(2, 3)
            train_val_group_idx, test_group_idx = torch.tensor(train_val_group_idx), torch.tensor(test_group_idx)

            def split_idx(group_idx):
                split_idx = torch.zeros(len(torch.squeeze(self._metadata_array_wo_y)), dtype=torch.bool)
                for idx in group_idx:
                    split_idx += (torch.squeeze(self._metadata_array_wo_y) == idx)
                return split_idx

            train_val_split_idx, test_split_idx = split_idx(train_val_group_idx), split_idx(test_group_idx)

            train_val_split_idx = torch.arange(dataset_size, dtype=torch.int64)[train_val_split_idx]
            train_val_sets_size = len(train_val_split_idx)
            random_index = np.random.permutation(train_val_sets_size)
            train_split_idx = train_val_split_idx[random_index[:int(3 / 4 * train_val_sets_size)]]
            val_split_idx = train_val_split_idx[random_index[int(3 / 4 * train_val_sets_size):]]

        self._split_array[train_split_idx] = 0
        self._split_array[val_split_idx] = 1
        self._split_array[test_split_idx] = 2

        if dataset_kwargs['model'] == '3wlgnn':
            self._collate = self.collate_dense
        else:
            if torch_geometric.__version__ >= '1.7.0':
                self._collate = PyGCollater(follow_batch=[], exclude_keys=[])
            else:
                self._collate = PyGCollater(follow_batch=[])

        self._metric = Evaluator('ogbg-molhiv')


        # GSN
        self.subgraph = subgraph
        if self.subgraph:
            self.id_type = dataset_kwargs['gsn_id_type']
            self.k = dataset_kwargs['gsn_k']
            from gds.datasets.gsn.gsn_data_prep import GSN
            subgraph = GSN(dataset_name='ColoredMNIST', dataset_group='MNIST', induced=True, id_type=self.id_type,
                           k=self.k)

            self.graphs_ptg, self.encoder_ids, self.d_id, self.d_degree = subgraph.preprocess('data/ColoredMNIST')

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
        # y_true = y_true.view(-1, 1)
        # y_pred = (y_pred > 0).long().view(-1, 1)
        # input_dict = {"y_true": y_true, "y_pred": y_pred}
        # acc = (y_pred == y_true).sum() / len(y_pred)
        # results = {'acc': np.float(acc)}

        # return results, f"Accuracy: {acc:.3f}\n"
        assert prediction_fn is None
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        results = self._metric.eval(input_dict)

        return results, f"ROCAUC: {results['rocauc']:.3f}\n"

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense(self, samples):
        # The input samples is a list of pairs (graph, label).

        graph_list, y_list, metadata_list = map(list, zip(*samples))
        y, metadata = torch.tensor(y_list), torch.stack(metadata_list)

        # insert size one at dim 0 because this dataset's y is 1d
        y = y.unsqueeze(0)

        x_node_feat = []
        for graph in graph_list:
            adj = self._sym_normalize_adj(to_dense_adj(graph.edge_index).squeeze())
            zero_adj = torch.zeros_like(adj)
            in_dim = graph.x.shape[1]
            # in_dim = 10

            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for _ in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)

            for node, node_feat in enumerate(graph.x):
                adj_node_feat[1:, node, node] = node_feat
                # adj_node_feat[1:3, node, node] = node_feat[:2]
                # adj_node_feat[3:, node, node] = F.one_hot(node_feat[2].long(), 8)

            x_node_feat.append(adj_node_feat)
        x_node_feat = torch.stack(x_node_feat)

        return x_node_feat, y, metadata

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
