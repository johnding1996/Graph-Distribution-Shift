import torch
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.data.dataloader import Collater as PyGCollater

from gds.datasets.wilds_dataset import GDSDataset


class OGBPROTEINSDataset(GDSDataset):
    """
    The OGB-Proteins dataset.
    This dataset is directly adopted from Open Graph Benchmark, and originally curated by MoleculeNet.

    Supported `split_scheme`:
        - 'species',


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

    _dataset_name = 'ogb-proteins'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', **dataset_kwargs):
        self._version = version
        if version is not None:
            raise ValueError('Versioning for OGB-Proteins is handled through the OGB package. Please set version=none.')
        # internally call ogb package
        self.ogb_dataset = PygNodePropPredDataset(name='ogbn-proteins', root=root_dir, transform=T.ToSparseTensor())

        # set variables
        self._data_dir = self.ogb_dataset.root
        if split_scheme == 'official':
            split_scheme = 'species'
        self._split_scheme = split_scheme
        self._y_type = 'float'  # although the task is binary classification, the prediction target contains nan value, thus we need float
        self._y_size = self.ogb_dataset.num_tasks
        self._n_classes = self.ogb_dataset.__num_classes__

        self._y_array = self.ogb_dataset.data.y
        self._split_array = torch.zeros(len(self._y_array)).long()
        split_idx = self.ogb_dataset.get_idx_split()

        self._split_array[split_idx['train']] = 0
        self._split_array[split_idx['valid']] = 1
        self._split_array[split_idx['test']] = 2

        # Move edge features to node features.
        data = self.ogb_dataset[0]
        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)

        self._metadata_fields = ['species']
        self._metadata_array = self.ogb_dataset.data.node_species.reshape(-1, 1).long()

        if torch_geometric.__version__ >= '1.7.0':
            self._collate = PyGCollater(follow_batch=[], exclude_keys=[])
        else:
            self._collate = PyGCollater(follow_batch=[])

        self._metric = Evaluator('ogbn-proteins')

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
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
        assert prediction_fn is None, "OGBHIVDataset.eval() does not support prediction_fn. Only binary logits accepted"
        input_dict = {"y_true": y_true, "y_pred": y_pred}

        results = self._metric.eval(input_dict)

        return results, f"ROCAUC: {results['rocauc']:.3f}\n"
