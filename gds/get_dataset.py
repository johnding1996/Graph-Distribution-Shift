import gds


def get_dataset(dataset, version=None, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        version (str): Dataset version number, e.g., '1.0'.
                       Defaults to the latest version.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in gds.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {gds.supported_datasets}.')

    if dataset == 'ogb-molpcba':
        from gds.datasets.ogbmolpcba_dataset import OGBPCBADataset
        return OGBPCBADataset(version=version, **dataset_kwargs)

    elif dataset == 'ogb-molhiv':
        from gds.datasets.ogbmolhiv_dataset import OGBHIVDataset
        return OGBHIVDataset(version=version, **dataset_kwargs)

    elif dataset == 'ogbg-ppa':
        from gds.datasets.ogbgppa_dataset import OGBGPPADataset
        return OGBGPPADataset(version=version, **dataset_kwargs)

    elif dataset == 'RotatedMNIST':
        from gds.datasets.rotated_mnist_dataset import RotatedMNISTDataset
        return RotatedMNISTDataset(version=version, **dataset_kwargs)

    elif dataset == 'SBM-Environment':
        from gds.datasets.sbm_environment_dataset import SBMEnvironmentDataset
        return SBMEnvironmentDataset(version=version, **dataset_kwargs)

    elif dataset == 'SBM-Isolation':
        from gds.datasets.sbm_isolation_dataset import SBMIsolationDataset
        return SBMIsolationDataset(version=version, **dataset_kwargs)

    elif dataset == 'UPFD':
        from gds.datasets.upfd_dataset import UPFDDataset
        return UPFDDataset(version=version, **dataset_kwargs)

    else:
        raise NotImplementedError
