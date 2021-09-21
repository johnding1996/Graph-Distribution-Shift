dataset_defaults = {
    'ogb-molpcba': {
        'num_domains': 120084,
        'split_scheme': 'official',
        'model_kwargs': {'dropout': 0.5, 'dataset_group':'mol'},  # include pretrained
        'default_frac': 0.10,
        'loss_function': 'multitask_bce',
        'groupby_fields': ['scaffold', ],
        'val_metric': 'ap',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 128,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 150,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multitask_binary_accuracy',
    },
    'ogb-molhiv': {
        'num_domains': 19089,
        'split_scheme': 'official',
        'model_kwargs': {'dropout':0.5, 'dataset_group':'mol'}, # include pretrained
        'default_frac': 1.0,
        'loss_function': 'BCEWithLogitsLoss',
        'groupby_fields': ['scaffold', ],
        'val_metric': 'rocauc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 128,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 150,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'binary_accuracy',
    },
    'ogbg-ppa': {
        'num_domains': 100000000000,
        'split_scheme': 'official',
        'model_kwargs': {'dropout':0.5, 'dataset_group':'ppa'}, # include pretrained
        'default_frac': 0.10,
        'loss_function': 'cross_entropy',
        'groupby_fields': ['species', ],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 16,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 150,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multiclass_accuracy',
    },
    'RotatedMNIST': {
        'num_domains': 6,
        'split_scheme': 'official',
        'model_kwargs': {'dropout':0.5, 'dataset_group':'RotatedMNIST'}, # include pretrained
        'default_frac': 1.0,
        'loss_function': 'cross_entropy',
        'groupby_fields': ['scaffold',],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 128,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 150,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multiclass_accuracy',
    },
    'SBM1': {
        'split_scheme': 'official',
        'model': 'gin_virtual_mnist',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'default_frac': 1.0,
        'loss_function': 'cross_entropy',
        'groupby_fields': ['scaffold',],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 128,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 150,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multiclass_accuracy',
    },
    'UPFD': {
        'num_domains': 10,
        'split_scheme': 'official',
        'model_kwargs': {'dropout':0.5, 'dataset_group':'mol'}, # include pretrained
        'default_frac': 1.0,
        'loss_function': 'BCEWithLogitsLoss',
        'groupby_fields': ['scaffold',],
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 128,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 150,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'binary_accuracy',
    },
}
