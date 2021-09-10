import wilds

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

    if dataset not in wilds.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {wilds.supported_datasets}.')


    if dataset == 'ogb-molpcba':
        from wilds.datasets.ogbmolpcba_dataset import OGBPCBADataset
        return OGBPCBADataset(version=version, **dataset_kwargs)

    elif dataset == 'ogb-molhiv':
        from wilds.datasets.ogbmolhiv_dataset import OGBHIVDataset
        return OGBHIVDataset(version=version, **dataset_kwargs)
    
    elif dataset == 'ogbg-ppa':
        from wilds.datasets.ogbgppa_dataset import OGBGPPADataset
        return OGBGPPADataset(version=version, **dataset_kwargs)

    elif dataset == 'ogb-proteins':
        from wilds.datasets.ogbproteins_dataset import OGBPROTEINSDataset
        return OGBPROTEINSDataset(version=version, **dataset_kwargs)

    elif dataset == 'RotatedMNIST':
        from wilds.datasets.superpixel_dataset import SuperPixelDataset
        return SuperPixelDataset(version=version, **dataset_kwargs)

    elif dataset == 'SBM1':
        from wilds.datasets.sbm_1_dataset import SBM1Dataset
        return SBM1Dataset(version=version, **dataset_kwargs)
