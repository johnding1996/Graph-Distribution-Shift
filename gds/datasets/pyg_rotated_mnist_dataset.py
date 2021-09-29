import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from typing import Optional, Callable, List


class PyGRotatedMNISTDataset(InMemoryDataset):
    names = ['RotatedMNIST', 'RotatedMNIST_expired']
    urls = {
        'RotatedMNIST': 'https://www.dropbox.com/s/3tgk65bpjp8mvhj/RotatedMNIST.zip?dl=1',
        'RotatedMNIST_expired': 'https://www.dropbox.com/s/5kybifusm8jexna/RotatedMNIST_expired.zip?dl=1'
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.names
        if name == 'RotatedMNIST':
            self.num_tasks = 1
            self.__num_classes__ = 10
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

        data_list = []
        for data_dict in inputs :
            data_dict['x'] = data_dict['x'][:, :1]
            data_list.append(Data(**data_dict))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
