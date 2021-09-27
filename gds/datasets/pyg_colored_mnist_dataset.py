import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from typing import Optional, Callable, List
import numpy as np


class PyGColoredMNISTDataset(InMemoryDataset):
    names = ['ColoredMNIST']
    urls = {
        'ColoredMNIST': 'https://www.dropbox.com/s/bz0mkww2ivu8rwn/ColoredMNIST.zip?dl=1'
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.names
        if name == 'ColoredMNIST':
            self.num_tasks = 1
            self.__num_classes__ = 1
        else:
            raise NotImplementedError

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
        return [f'{self.name}_processed_mean_px_feat.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        inputs = torch.load(self.raw_paths[0])
        # data_list = [Data(**data_dict) for data_dict in inputs]

        total_num_nodes = 0
        for data_dict in inputs :
            total_num_nodes += data_dict['x'].shape[0]

        np.random.seed(0)
        padded_cate = np.random.randint(8, size=total_num_nodes)

        data_list = []
        ptr = 0
        for i, data_dict in enumerate(inputs) :
            num_node = data_dict['x'].shape[0]
            cate = torch.tensor(padded_cate[ptr:ptr+num_node]).view((-1,1))
            x = torch.cat((data_dict['x'][:, :2], cate), dim=1)
            data_dict['x'] = x
            data_list.append(Data(**data_dict))
            ptr += num_node
        assert ptr == total_num_nodes

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
