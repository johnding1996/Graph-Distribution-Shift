from .get_dataset import get_dataset
from .version import __version__

benchmark_datasets = [
    'ogb-molpcba',
    'ogb-molhiv',
    'ogbg-ppa',
    'ogb-proteins',
    'RotatedMNIST',
    'SBM1'
]

supported_datasets = benchmark_datasets