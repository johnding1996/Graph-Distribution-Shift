import os
import os.path as osp
from typing import Optional, Callable, List

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
import os


class PyGSuperPixelDataset(InMemoryDataset):
    r"""A variety of artificially and semi-artificially generated graph
    datasets from the `"Benchmarking Graph Neural Networks"
    <https://arxiv.org/abs/2003.00982>`_ paper.

    .. note::
        The ZINC dataset is provided via
        :class:`torch_geometric.datasets.ZINC`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    names = ['RotatedMNIST']
    urls = {
        'RotatedMNIST': 'https://www.dropbox.com/s/5kybifusm8jexna/RotatedMNIST.zip?dl=1'
        # 'https://www.dropbox.com/s/v2iwdes7k3vuk4l/RotatedMNIST.pt?dl=1'
    }



    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.names
        if name == 'RotatedMNIST' :
             self.num_tasks = 1
             self.__num_classes__ = 10
        else :
            raise NotImplemented

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # raw file name must match the dataset name
        return [f'{self.name}.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.name}_processed.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        inputs = torch.load(self.raw_paths[0])

        data_list = [Data(**data_dict) for data_dict in inputs]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]


        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == '__main__':
    root = '/home/ubuntu/Graph-Distribution-Shift/preprocessing/superpixel/data'
    name = 'RotatedMNIST'
    dataset = OurDataset(root=root, name=name)

    # pdb.set_trace()

    self._y_size = 1
    self._n_classes = 10
    self._metric = Evaluator('ogbg-ppa')
