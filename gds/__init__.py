from .get_dataset import get_dataset
from .version import __version__

benchmark_datasets = [
    'ogb-molpcba',
    'ogb-molhiv',
    'ogbg-ppa',
    'RotatedMNIST',
    'ColoredMNIST',
    'SBM-Environment',
    'SBM-Isolation',
    'UPFD'
]

supported_datasets = benchmark_datasets
