import os
from .utils_graph_processing import subgraph_isomorphism_edge_counts, subgraph_isomorphism_vertex_counts, \
    induced_edge_automorphism_orbits, edge_automorphism_orbits, automorphism_orbits
from .utils_ids import subgraph_counts2ids
from .utils_data_gen import generate_dataset
from .utils_graph_learning import multi_class_accuracy
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.data import Data
import glob
import re
import types


def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    '''
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))

    return edge_lists


def prepare_dataset(path,
                    dataset,
                    name,
                    id_scope,
                    id_type,
                    k,
                    extract_ids_fn,
                    count_fn,
                    automorphism_fn,
                    multiprocessing,
                    num_processes,
                    **subgraph_params):
    if dataset in ['bioinformatics', 'social', 'chemical', 'ogb', 'SR_graphs', 'MNIST', 'SBM', 'UPFD']:
        data_folder = os.path.join(path, 'processed', id_scope)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        if id_type != 'custom':
            if subgraph_params['induced']:
                if subgraph_params['directed_orbits'] and id_scope == 'local':
                    data_file = os.path.join(data_folder, '{}_induced_directed_orbits_{}.pt'.format(id_type, k))
                else:
                    data_file = os.path.join(data_folder, '{}_induced_{}.pt'.format(id_type, k))
            else:
                if subgraph_params['directed_orbits'] and id_scope == 'local':
                    data_file = os.path.join(data_folder, '{}_directed_orbits_{}.pt'.format(id_type, k))
                else:
                    data_file = os.path.join(data_folder, '{}_{}.pt'.format(id_type, k))
            maybe_load = True
        else:
            data_file = None  # we don't save custom substructure counts
            maybe_load = False
        loaded = False
    else:
        raise NotImplementedError("Dataset family {} is not currently supported.".format(dataset))

    # try to load, possibly downgrading
    if maybe_load:

        if os.path.exists(data_file):  # load
            graphs_ptg, num_classes, orbit_partition_sizes = load_dataset(data_file)
            loaded = True

        else:  # try downgrading. Currently works only when for each k there is only one substructure in the family
            if id_type in ['cycle_graph',
                           'path_graph',
                           'complete_graph',
                           'binomial_tree',
                           'star_graph']:
                k_min = 2 if id_type == 'star_graph' else 3
                succeded, graphs_ptg, num_classes, orbit_partition_sizes = try_downgrading(data_folder,
                                                                                           id_type,
                                                                                           subgraph_params['induced'],
                                                                                           subgraph_params[
                                                                                               'directed_orbits']
                                                                                           and id_scope == 'local',
                                                                                           k, k_min)
                if succeded:  # save the dataset
                    print("Saving dataset to {}".format(data_file))
                    torch.save((graphs_ptg, num_classes, orbit_partition_sizes), data_file)
                    loaded = True

    if not loaded:
       
        graphs_ptg, num_classes, num_node_type, num_edge_type, orbit_partition_sizes = generate_dataset(path,
                                                                                                        name,
                                                                                                        k,
                                                                                                        extract_ids_fn,
                                                                                                        count_fn,
                                                                                                        automorphism_fn,
                                                                                                        id_type,
                                                                                                        multiprocessing,
                                                                                                        num_processes,
                                                                                                        **subgraph_params)
        if data_file is not None:
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_ptg, num_classes, orbit_partition_sizes), data_file)

        if num_node_type is not None:
            torch.save((num_node_type, num_edge_type), os.path.join(path, 'processed', 'num_feature_types.pt'))

    return graphs_ptg, num_classes, orbit_partition_sizes


def load_dataset(data_file):
    '''
        Loads dataset from `data_file`.
    '''
    print("Loading dataset from {}".format(data_file))
    dataset_obj = torch.load(data_file)
    graphs_ptg = dataset_obj[0]
    num_classes = dataset_obj[1]
    orbit_partition_sizes = dataset_obj[2]

    return graphs_ptg, num_classes, orbit_partition_sizes


def try_downgrading(data_folder, id_type, induced, directed_orbits, k, k_min):
    '''
        Extracts the substructures of size up to the `k`, if a collection of substructures
        with size larger than k has already been computed.
    '''
    found_data_filename, k_found = find_id_filename(data_folder, id_type, induced, directed_orbits, k)
    if found_data_filename is not None:
        graphs_ptg, num_classes, orbit_partition_sizes = load_dataset(found_data_filename)
        print("Downgrading k from dataset {}...".format(found_data_filename))
        graphs_ptg, orbit_partition_sizes = downgrade_k(graphs_ptg, k, orbit_partition_sizes, k_min)
        return True, graphs_ptg, num_classes, orbit_partition_sizes
    else:
        return False, None, None, None


def find_id_filename(data_folder, id_type, induced, directed_orbits, k):
    '''
        Looks for existing precomputed datasets in `data_folder` with counts for substructure 
        `id_type` larger `k`.
    '''
    if induced:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_induced_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_induced_[0-9]*.pt'.format(id_type))
    else:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_[0-9]*.pt'.format(id_type))
    filenames = glob.glob(pattern)
    for name in filenames:
        k_found = int(re.findall(r'\d+', name)[-1])
        if k_found >= k:
            return name, k_found
    return None, None

def downgrade_k(dataset, k, orbit_partition_sizes, k_min):
    '''
        Donwgrades `dataset` by keeping only the orbits of the requested substructures.
    '''
    feature_vector_size = sum(orbit_partition_sizes[0:k - k_min + 1])
    graphs_ptg = list()
    for data in dataset:
        new_data = Data()
        for attr in data.__iter__():
            name, value = attr
            setattr(new_data, name, value)
        setattr(new_data, 'identifiers', data.identifiers[:, 0:feature_vector_size])
        graphs_ptg.append(new_data)
    return graphs_ptg, orbit_partition_sizes[0:k - k_min + 1]
