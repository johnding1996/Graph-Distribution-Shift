algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
    },
    'groupDRO': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'group_dro_step_size': 0.01,
    },
    'deepCORAL': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'coral_penalty_weight': 0.1,
    },
    'IRM': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'irm_lambda': 1.,
        'irm_penalty_anneal_iters': 500,
    },
    'FLAG': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'flag_step_size': 1e-3
    },
    'GSN': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
    },
    'DANN': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'dann_lambda': 1.
    },
    'CDANN': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'dann_lambda': 1.
    },
    'DANN-G': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'dann_lambda': 1.
    },
    'MLDG': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'mldg_beta': 1.
    },
}
