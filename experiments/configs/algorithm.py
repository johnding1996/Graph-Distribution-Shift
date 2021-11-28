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
        'gcl_contrastive_pretrain': False, #True, # False would be what was run before, ERM+aug
        'gcl_reinit_optim_sched': True, # False is just naively swap the losses and continue
        'gcl_finetune_lr_scale':  1.0, #1.0, if 'reinit'=True, downscale ratio for lr during finetune, 1.0=same as pretrain, for Adam this probs doesn't matter
        'gcl_finetune_feat_cls_scale': 0.2, # implm TODO, ratio of featurizer lr to classifier lr, 0.0 = frozen featurizer, 0.2 = 1:5 feat:classifier tuning, 1.0 = 1:1
        'gcl_pretrain_fraction': 0.5, # 0.0, would mean no pretraining phase, 0.5 is 100 of 200 epochs
        'gcl_contrast_type': 'opposite', # 'orig', 'opposite', 'random', relative to 'aug_type'
        'gcl_aug_prob': 1.0, #0.0 should be equiv to ERM, as aug is never applied, 1.0 is GCL setting where orig data is never used
        # 'aug_type': 'node_drop', #'node_drop','edge_perm','random' can only use a single aug_type currently, or random choice of the two
        'gcl_aug_type': 'node_drop',
        'gcl_aug_ratio': 0.2 #0.0 should be equiv to ERM, as each graph is unchanged, 0.2 is GCL paper default
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
