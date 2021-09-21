import torch.nn as nn
from torch_geometric.nn import global_mean_pool

<<<<<<< HEAD

def initialize_model(config, d_out, full_dataset=None, is_featurizer=False):
=======
def initialize_model(config, d_out, is_featurizer=False, is_pooled=True):
>>>>>>> master
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

<<<<<<< HEAD
    if config.gsn == True:
        from models.gsn.gnn import GNN_OGB
        model = GNN_OGB(in_features=full_dataset.num_features,
                out_features=d_out, 
                encoder_ids=full_dataset.encoder_ids,
                d_in_id=full_dataset.d_id,
                in_edge_features=full_dataset.num_edge_features, 
                d_in_node_encoder=full_dataset.d_in_node_encoder,
                d_in_edge_encoder=full_dataset.d_in_edge_encoder,
                d_degree=full_dataset.d_degree)
    else:
        from models.gnn import GNN
        if is_featurizer:
            featurizer = GNN(gnn_type=config.model, num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GNN(gnn_type=config.model, num_tasks=d_out, **config.model_kwargs)


=======

    from models.gnn import GNN
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
>>>>>>> master


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
