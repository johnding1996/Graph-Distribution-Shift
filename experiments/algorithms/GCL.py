from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import torch
from torch_geometric.data import Batch
from torch.nn.utils import clip_grad_norm_
from utils import move_to

class GCL(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # If ratio is 0.0, should be equiv to ERM
        self.aug_type = config.aug_type
        self.aug_ratio = config.aug_ratio

    def drop_nodes(self, data):
        """
        https://github.com/Shen-Lab/GraphCL/blob/1d43f79d7f33f8133f9d4b4b8254d8aaeb09a615/semisupervised_TU/pre-training/tu_dataset.py#L139
        Original method operates on 
        the adjacency matrix like so

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        ...
        edge_index = data.edge_index.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj[idx_nondrop, :][:, idx_nondrop]
        edge_index = adj.nonzero().t()

        This is nice, and pretty efficient, but here we have
        to directly operate on the node index+edge lists, 
        to be careful to keep the edge attributes aligned
        """

        aug_ratio = self.aug_ratio
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()

        # Directly model the uniform drop prob over nodes
        drop_num = int(node_num  * aug_ratio)
        idx_perm = torch.randperm(node_num)
        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:].sort().values.cuda() # sort for humans/debug

        # Realize this ^ as a subselecting of the edges in the graph,
        # Noting that this isn't an elegant process because we need to 
        # preserve the orig edge attributes
        orig_edge_index = data.edge_index

        src_subselect = torch.nonzero(idx_nondrop[...,None] == orig_edge_index[0])[:,1]
        dest_subselect = torch.nonzero(idx_nondrop[...,None] == orig_edge_index[1])[:,1]
        edge_subselect = src_subselect[torch.nonzero(src_subselect[..., None] == dest_subselect)[:,0]].sort().values

        new_edge_index = orig_edge_index[:,edge_subselect]
        _, new_edge_num = new_edge_index.size()

        if data.edge_attr is not None:
            orig_edge_attr = data.edge_attr
            new_edge_attr = orig_edge_attr[edge_subselect] 

        # assumption is we are removing undirected edge pairs i<->j
        assert (edge_num-new_edge_num)%2 is 0

        try:
            data.edge_attr = new_edge_attr
            data.edge_index = new_edge_index
            # data.x =  ... We do not modify the node features
        except:
            pass
            # data = data

    # Other augmentation types, TODO
    def weighted_drop_nodes(self, data):
        """
        same as drop nodes but instead of uniform drop over nodes
        weighting of drop by degree of node
        """
        raise NotImplementedError
    def permute_edges(self, data):

        """
        Ported from same repo as drop_nodes

        Current: 
        In-place replacement k edges (i->j) with k edges (i'->j') 
        where i and j are uniformly generated over the node idx, i' != j'

        TODO 1:
        Currently the edge_attr of (i->j) in inherited by (i'->j') 
        for datasets that include edge_attr, this is not very 
        semantically sound, as the joint distribution between 
        (i_feat,j_feat,e_attr), i_feat,j_feat in {node_features}, 
        e_attr in {edge_attr_types}, is not being respected

        TODO 2:
        A non-naive, 'paired' version will also rely on assumption 
        that this is a default pytorch geometric data object where 
        edge ordering is such that two the directions of a single 
        edge occur in pairs. So that we can operate on bidirectional 
        edges easily by stride-2 indexing
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])

        """
        aug_ratio = self.aug_ratio
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()

        paired_perms = False # for later, TODO
        if paired_perms: assert edge_num%2 is 0 # undirected g, paired edges

        permute_num = int(edge_num * aug_ratio)

        orig_edge_index = data.edge_index

        if permute_num > 0:
            edges_to_insert = torch.multinomial(torch.ones(permute_num,node_num), 2, replacement=False).t().cuda() 
            insertion_indices =  torch.multinomial(torch.ones(edge_num), permute_num, replacement=False)

            if paired_perms:
                raise NotImplementedError            

            orig_edge_index[:,insertion_indices] = edges_to_insert
        else:
            # augmentation silently does nothing
            pass
        try:
            data.edge_index = orig_edge_index #modified in place
        except:
            pass

    def extract_subgraph(self, data):
        """
        TODO
        paper uses a random walk to generate a subgraph of nodes
        """
        raise NotImplementedError

    def update(self, batch):

        assert self.is_training
        # process batch

        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata,
        }

        # Augmentation options
        # currently singular, TODO make composeable
        if self.aug_type == 'node_drop':
            aug_fn = self.drop_nodes
        elif self.aug_type == 'edge_perm':
            aug_fn = self.permute_edges
        else:
            raise NotImplementedError

        # Unpack batch of graphs and apply augmentation 
        # to each graph, in place
        graphs = x.to_data_list()
        for i in range(len(graphs)): 
            aug_fn(graphs[i])
        x = Batch.from_data_list(graphs)
        
        # Continue as with other methods/ERM
        outputs = self.model(x)
        results['y_pred'] = outputs

        objective = self.objective(results)
        results['objective'] = objective.item()

        self.optimizer.zero_grad()
        self.model.zero_grad()
        objective.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)

        # log results
        self.update_log(results)
        return self.sanitize_dict(results)

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
