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
        'coral_penalty_weight': 1.,
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
        'flag_step_size': 1e-2
    },
    'GCL': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'use_cl': False,
        'gcl_aug_prob': 0.5, #0.0 should be equiv to ERM, as aug is never applied, 1.0 is GCL setting where orig data is never seen
        'aug_type': 'random', #'node_drop','edge_perm','random' can only use a single aug_type currently, or random choice of them
        'gcl_aug_ratio': 0.1 # 0.0 should be equiv to ERM, as each graph is unchanged, 0.2 is GCL paper default
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
        'dann_lambda': 0.1
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
        'dann_lambda': 0.1
    },
    'MLDG': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'mldg_beta': 0.1
    },
}
