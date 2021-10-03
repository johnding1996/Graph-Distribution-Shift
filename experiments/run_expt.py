import argparse
import csv
import os
import sys
import time
from collections import defaultdict

try:
    import graph_tool
except Exception as e:
    pass
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision

import configs.supported as supported
import gds
from algorithms.initializer import initialize_algorithm
from configs.utils import populate_defaults
from train import train, evaluate
from utils import set_seed, Logger, BatchLogger, log_config, initialize_wandb, close_wandb, ParseKwargs, load, \
    log_group_data, parse_bool, get_model_prefix
from gds.common.data_loaders import get_train_loader, get_eval_loader
from gds.common.grouper import CombinatorialGrouper



def main():
    """ to see default hyperparams for each dataset/model, look at configs/ """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=gds.supported_datasets, required=True)
    parser.add_argument('-a', '--algorithm', choices=supported.algorithms, required=True)
    parser.add_argument('-m', '--model', choices=supported.models)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--use_frac', type=parse_bool,
                help='Convenience parameter that scales all dataset splits down to the specified fraction, '
                     'for development purposes. Note that this also scales the test set down, so the reported '
                     'numbers are not comparable with the full test set.')
    
   
    # Resume
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False)

    # Dataset
    parser.add_argument('--split_scheme',
                        help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to downloads the dataset if it does not exist in root_dir.')
    parser.add_argument('--version', default=None, type=str)

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')

    # Model
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    ## To be tuned
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--flag_step_size', type=float)
    parser.add_argument('--dann_lambda', type=float)
    parser.add_argument('--mldg_beta', type=float)
    parser.add_argument('--gcl_aug_ratio', type=float)
    parser.add_argument('--parameter', type=float)

    ## Not to be tuned
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--gsn_id_type', type=str,
                        choices=['cycle_graph', 'path_graph', 'complete_graph', 'binomial_tree', 'nonisomorphic_trees'

])
    parser.add_argument('--gsn_k', type=int)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={})

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--process_outputs_function', choices=supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_epoch', default=None, type=int,
                        help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Ablation
    parser.add_argument('--random_split', type=parse_bool, const=True, nargs='?', default=False)

    # Misc
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--root_dir', default='./data',
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)

    parser.add_argument('--jiuhai', action='store_true', default=False)

    config = parser.parse_args()
    config = populate_defaults(config)

    
    config.suffix = str(time.time())[-3:]

    

    # hard code:
    if config.jiuhai :
        if config.dataset in {'ogb-molpcba', 'ogbg-ppa'} :
            if config.algorithm in {'deepCORAL', 'FLAG', 'GCL'} or config.model in {'gin_10_layers', 'cheb'} :
                raise ValueError('For Jiuhai\'s experiments, these are too slow, kill.')

    if config.algorithm == 'MLDG' :
        assert config.dataset != 'ogb-molpcba' and config.dataset != 'ogbg-ppa'

    if config.model == 'cheb' and config.algorithm == 'GCL' :
        config.gcl_aug_ratio = 0.1

    # To speed up slow algorithms
    if (config.algorithm == 'MLDG' or config.algorithm == 'FLAG') and config.dataset != 'SBM-Isolation' :
        config.n_epochs = config.n_epochs//2
    if config.algorithm == 'DANN' or config.algorithm == 'DANN-G' :
        config.n_epochs *= 2
    if config.algorithm == 'IRM' :
        config.n_epochs = int(config.n_epochs * 1.5)

    if config.algorithm == 'deepCORAL':
        config.parameter = config.coral_penalty_weight
    elif config.algorithm == 'DANN' or config.algorithm == 'DANN-G':
        config.parameter = config.dann_lambda
    elif config.algorithm == 'MLDG':
        config.parameter = config.mldg_beta
    elif config.algorithm == 'IRM':
        config.parameter = config.irm_lambda
    elif config.algorithm == 'FLAG':
        config.parameter = config.flag_step_size
    elif config.algorithm == 'GCL':
        config.parameter = config.gcl_aug_ratio
    else:
        config.parameter = None

    
    # For the 3wlgnn model, we need to set batch_size to 1
    if config.model == '3wlgnn':
        config.batch_size = 1

    # Set device
    config.device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume = True
        mode = 'a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume = False
        mode = 'a'
    else:
        resume = False
        mode = 'w'


    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, f'{config.dataset}_{config.algorithm}_{config.parameter}_{config.model}_seed-{config.seed}_{config.suffix}.txt'), mode)


    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    if config.algorithm == 'GSN':
        config.dataset_kwargs['gsn_id_type'] = config.gsn_id_type
        config.dataset_kwargs['gsn_k'] = config.gsn_k
    full_dataset = gds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        random_split=config.random_split,
        subgraph=True if config.algorithm == 'GSN' else False,
        algorithm=config.algorithm,
        model=config.model,
        **config.dataset_kwargs)

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields)

    datasets = defaultdict(dict)
    if config.use_wandb:
        wandb_runner = initialize_wandb(config)
    for split in full_dataset.split_dict.keys():
        if split == 'train':
            verbose = True
        elif split == 'val':
            verbose = True
        else:
            verbose = False

        # Get subset
        if config.use_frac:
            datasets[split]['dataset'] = full_dataset.get_subset(split, frac=config.default_frac)
        else:
            datasets[split]['dataset'] = full_dataset.get_subset(split, frac=1.0)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{config.dataset}_{config.algorithm}_{config.parameter}_{config.model}_seed-{config.seed}_{split}_eval_{config.suffix}.csv'), mode=mode, use_wandb=(config.use_wandb and verbose))
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{config.dataset}_{config.algorithm}_{config.parameter}_{config.model}_seed-{config.seed}_{split}_algo_{config.suffix}.csv'), mode=mode, use_wandb=(config.use_wandb and verbose))

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size == 1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    
    ## Initialize algorithm
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        full_dataset=full_dataset,
        train_grouper=train_grouper)

    model_prefix = get_model_prefix(datasets['train'], config)
   

    if not config.eval_only:
        ## Load saved results if resuming
        resume_success = False
        if resume:
            save_path = model_prefix.with_name(model_prefix.name + 'epoch-last_model.pth')
            if not os.path.exists(save_path):
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = model_prefix.with_name(model_prefix.name + f'epoch-{latest_epoch}_model.pth')
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass

        if resume_success == False:
            epoch_offset = 0
            best_val_metric = None
        start = time.time()
        train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            result_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric)
    else:
        if config.eval_epoch is None:
            eval_model_path = model_prefix.with_name(model_prefix.name + 'epoch-best_model.pth')
        else:
            eval_model_path = model_prefix.with_name(model_prefix.name + f'epoch-{config.eval_epoch}_model.pth')
        best_epoch, best_val_metric = load(algorithm, eval_model_path)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        if epoch == best_epoch:
            is_best = True
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            result_logger=logger,
            config=config,
            is_best=is_best)

    # have to close wandb runner before closing logger (and stdout)
    if config.use_wandb:
        close_wandb(wandb_runner)
    finish = time.time()
    if not config.eval_only:
        logger.write(f'time(s): {finish-start:.3f}\n')
    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()


if __name__ == '__main__':
    main()
