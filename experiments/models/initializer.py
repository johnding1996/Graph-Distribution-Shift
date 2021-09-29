import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.gnn import GNN
from models.three_wl import ThreeWLGNNNet
from models.gsn.gnn import GNN_GSN
from models.mlp import MLP


def initialize_model(config, d_out, is_featurizer=False, full_dataset=None, is_pooled=True):
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
    import pdb
    pdb.set_trace()

    if full_dataset is None:
        if config.model == "3wlgnn":
            if is_featurizer:
                featurizer = ThreeWLGNNNet(gnn_type=config.model, num_tasks=None, **config.model_kwargs)
                classifier = nn.Linear(featurizer.d_out, d_out)
                model = (featurizer, classifier)
            else:
                model = ThreeWLGNNNet(gnn_type=config.model, num_tasks=d_out, **config.model_kwargs)
        elif config.model == 'mlp' :
            assert config.algorithm == 'ERM' or config.algorithm == 'IRM' # combinations with other algorithms not checked
            if is_featurizer:
                featurizer = MLP(gnn_type=config.model, num_tasks=None, **config.model_kwargs)
                classifier = nn.Linear(featurizer.d_out, d_out)
                model = (featurizer, classifier)
            else:
                model = MLP(gnn_type=config.model, num_tasks=d_out, **config.model_kwargs)
        else:
            if is_featurizer:
                if is_pooled :
                    featurizer = GNN(gnn_type=config.model, num_tasks=None, is_pooled=is_pooled, **config.model_kwargs)
                    classifier = nn.Linear(featurizer.d_out, d_out)
                    model = (featurizer, classifier)
                else :
                    featurizer = GNN(gnn_type=config.model, num_tasks=None, is_pooled=is_pooled, **config.model_kwargs)
                    classifier = nn.Linear(featurizer.d_out, d_out)
                    pooler = global_mean_pool
                    model = (featurizer, pooler, classifier)
            else:
                model = GNN(gnn_type=config.model, num_tasks=d_out, is_pooled=is_pooled, **config.model_kwargs)

    # We use the full dataset only for GSN
    # Need to be refactored
    else:
        if is_featurizer:
            featurizer = GNN_GSN(in_features=full_dataset.num_features,
                                 out_features=None,
                                 encoder_ids=full_dataset.encoder_ids,
                                 d_in_id=full_dataset.d_id,
                                 in_edge_features=full_dataset.num_edge_features,
                                 d_in_node_encoder=full_dataset.d_in_node_encoder,
                                 d_in_edge_encoder=full_dataset.d_in_edge_encoder,
                                 d_degree=full_dataset.d_degree,
                                 dataset_group=config.model_kwargs['dataset_group'])
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GNN_GSN(in_features=full_dataset.num_features,
                            out_features=d_out,
                            encoder_ids=full_dataset.encoder_ids,
                            d_in_id=full_dataset.d_id,
                            in_edge_features=full_dataset.num_edge_features,
                            d_in_node_encoder=full_dataset.d_in_node_encoder,
                            d_in_edge_encoder=full_dataset.d_in_edge_encoder,
                            d_degree=full_dataset.d_degree,
                            dataset_group=config.model_kwargs['dataset_group'])

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
