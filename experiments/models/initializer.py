import torch.nn as nn


def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)
    """

    if config.model == 'gin_virtual_mol':
        from models.gnn_mol import GINVirtual_mol
        if is_featurizer:
            featurizer = GINVirtual_mol(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual_mol(num_tasks=d_out, **config.model_kwargs)


    elif config.model == 'gin_virtual_ppa':
        from models.gnn_ppa import GINVirtual_ppa
        if is_featurizer:
            featurizer = GINVirtual_ppa(num_class=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual_ppa(num_class=d_out, **config.model_kwargs)

    elif config.model == 'gin_virtual_mnist':
        from models.gnn_mnist import GINVirtual_mnist
        if is_featurizer:
            featurizer = GINVirtual_mnist(num_class=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual_mnist(num_class=d_out, **config.model_kwargs)

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if isinstance(model, tuple):
            for submodel in model:
                submodel.needs_y = False
        else:
            model.needs_y = False

    return model
