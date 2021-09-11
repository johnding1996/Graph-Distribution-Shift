from gds import supported_datasets
from experiments.configs.datasets import dataset_defaults


def test_datasets_match_gds():
    assert set(dataset_defaults.keys()) == set(supported_datasets)
