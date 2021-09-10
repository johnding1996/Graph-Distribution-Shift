import os

import pandas as pd
import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data.dataloader import Collater as PyGCollater

from gds.datasets.wilds_dataset import GDSDataset


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


class OGBGPPADataset(GDSDataset):
    """
    The OGB-ppa dataset.
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

    _dataset_name = 'ogbg-ppa'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        if version is not None:
            raise ValueError('Versioning for OGB-PPA is handled through the OGB package. Please set version=none.')
        # internally call ogb package
        self.ogb_dataset = PygGraphPropPredDataset(name='ogbg-ppa', root=root_dir, transform=add_zeros)

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
        self._split_array[split_idx['train']] = 0
        self._split_array[split_idx['valid']] = 1
        self._split_array[split_idx['test']] = 2

        self._y_array = self.ogb_dataset.data.y.squeeze()

        self._metadata_fields = ['species', 'y']
        metadata_file_path = os.path.join(self.ogb_dataset.root, 'mapping', 'graphidx2speciesid.csv.gz')
        df = pd.read_csv(metadata_file_path)
        df_np = df['species id'].to_numpy()
        self._metadata_array_wo_y = torch.from_numpy(df_np - df_np.min()).reshape(-1, 1).long()
        self._metadata_array = torch.cat((self._metadata_array_wo_y, self.ogb_dataset.data.y), 1)

        if torch_geometric.__version__ >= '1.7.0':
            self._collate = PyGCollater(follow_batch=[], exclude_keys=[])
        else:
            self._collate = PyGCollater(follow_batch=[])

        self._metric = Evaluator('ogbg-ppa')

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
        y_true = y_true.view(-1, 1)
        y_pred = torch.argmax(y_pred.detach(), dim=1).view(-1, 1)
        input_dict = {"y_true": y_true, "y_pred": y_pred}

        results = self._metric.eval(input_dict)

        return results, f"Accuracy: {results['acc']:.3f}\n"
