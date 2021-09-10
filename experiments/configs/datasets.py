dataset_defaults = {
    'ogb-molpcba': {
        'split_scheme': 'official',
        'model': 'gin_virtual_mol',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'frac': 0.1,
        'loss_function': 'multitask_bce',
        'groupby_fields': ['scaffold',],
        'val_metric': 'ap',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multitask_binary_accuracy',
    },
    'ogb-molhiv': {
        'split_scheme': 'official',
        'model': 'gin_virtual_mol',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'BCEWithLogitsLoss',
        'groupby_fields': ['scaffold',],
        'val_metric': 'rocauc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'binary_accuracy',
    },
    'ogbg-ppa': {
        'split_scheme': 'official',
        'model': 'gin_virtual_ppa',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'cross_entropy',
        'groupby_fields': ['species',],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multiclass_accuracy',
    },
    'ogb-proteins': {
        'split_scheme': 'official',
        'model': 'gin-virtual',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'multitask_bce',
        'groupby_fields': ['species',],
        'val_metric': 'rocauc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multitask_binary_accuracy',
    },
    'RotatedMNIST': {
        'split_scheme': 'official',
        'model': 'gin_virtual_mnist',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'cross_entropy',
        'groupby_fields': ['scaffold',],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multiclass_accuracy',
    },
    'SBM1': {
        'split_scheme': 'official',
        'model': 'gin_virtual_mol',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'cross_entropy',
        'groupby_fields': ['scaffold',],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multiclass_accuracy',
    }
}

