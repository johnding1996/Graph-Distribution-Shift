from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import torch
from torch_geometric.data import Batch
from torch.nn.utils import clip_grad_norm_
from utils import move_to

import math
from torch_geometric.utils import to_undirected

class GCL(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps):
        # model = initialize_model(config, d_out).to(config.device)
        featurizer, projector, classifier, combined_model = initialize_model(config, d_out, is_featurizer=True, include_projector=True)
        featurizer = featurizer.to(config.device)
        projector = projector.to(config.device)
        classifier = classifier.to(config.device)
        model = combined_model.to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.use_cl = config.use_cl
        self.aug_prob = config.aug_prob
        self.aug_type = config.aug_type
        # If use_cl=False and aug_ratio is 0.0, should be equiv to ERM
        self.aug_ratio = config.aug_ratio
        # set model components
        self.featurizer = featurizer
        self.projector = projector
        self.classifier = classifier
        # self.model = model #set by base class I think

        # additional logging, copied from deepCORAL, 
        # but throws error at end of epoch bc its "missing"
        # self.logged_fields.append('gsimclr_loss')

    def drop_nodes(self, data):
        #print("Node Drop!")
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
        node_num = data.x.size()[0]
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

        # This would only hold for undirected graph datasets
        # where the assumption is we are removing undirected edge pairs i<->j
        # assert (edge_num-new_edge_num)%2 is 0

        try:
            data.edge_attr = new_edge_attr
            data.edge_index = new_edge_index
            # data.x =  ... We do not modify the node features
            # breakpoint()
        except:
            pass
            # data = data

    def permute_edges(self, data):
        #print("Edge Permute!")
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
        node_num = data.x.size()[0]
        _, edge_num = data.edge_index.size()

        paired_perms = True #False # for later, TODO
        if paired_perms: 
            assert edge_num%2 is 0 # undirected g, paired edges

        permute_num = int(edge_num * aug_ratio)
        if paired_perms: 
           permute_num = 2 * math.ceil(permute_num / 2)

        orig_edge_index = data.edge_index

        if permute_num > 0:
            good_draws = 0
            while good_draws < permute_num:
                edges_to_insert = torch.multinomial(torch.ones(permute_num,node_num), 2, replacement=False).t().cuda()
                orig_edges_to_insert = edges_to_insert.clone().detach()
                edges_to_insert = edges_to_insert.unique(dim=1)
                w_reversed_edges = torch.cat((orig_edges_to_insert,torch.stack((orig_edges_to_insert[1],orig_edges_to_insert[0]))),dim=1)
                unique_pairs, rev_to_cts, cts = w_reversed_edges.unique(return_counts=True,return_inverse=True,dim=1)
                non_dupe = w_reversed_edges[:,:permute_num][:,cts[rev_to_cts[:permute_num]]<=1]
                good_draws = non_dupe.size()[1]
                edges_to_insert = non_dupe
            # single_copies = w_reversed_edges[:,:permute_num][:,cts[rev_to_cts[:permute_num]]>1].unique(dim=1)
            # edges_to_insert = torch.cat((non_dupe,single_copies),dim=1)
            
            # insertion_indices =  torch.multinomial(torch.ones(edge_num), permute_num, replacement=False)
            # breakpoint()
            src_nodes_match = torch.nonzero(edges_to_insert[0][...,None] == orig_edge_index[0])
            dst_nodes_match = torch.nonzero(edges_to_insert[1][...,None] == orig_edge_index[1])
            comb_nodes = torch.cat((src_nodes_match, dst_nodes_match))
            unique_idx_pairs, match_cts = comb_nodes.unique(return_counts=True,dim=0)
            both_match = unique_idx_pairs[match_cts > 1]
            match_mask = torch.ones(edges_to_insert.size()[1]).bool()
            match_mask[both_match[:,0]] = False
            edges_to_insert = edges_to_insert[:, match_mask]
            # if both_match.size()[0] > 0: breakpoint()
            reduced_permute_num = edges_to_insert.size()[1]

            mask_idx = torch.arange(start=1,end=edge_num, step=2)
            nat_nums = torch.ones(edge_num)
            nat_nums[mask_idx] = False # only evens
            evens = nat_nums
            orig_insertion_indices =  torch.multinomial(evens, reduced_permute_num, replacement=False)

            if paired_perms:
                src_nodes, dst_nodes = edges_to_insert[0], edges_to_insert[1]
                duped_src_nodes, duped_dst_nodes = src_nodes.repeat_interleave(2), dst_nodes.repeat_interleave(2) # torch.arange(start=1,end=48, step=2)
                swap_mask = torch.arange(start=1,end=reduced_permute_num*2, step=2)
                # if both_match.size()[0] > 0: breakpoint()
                duped_src_nodes[swap_mask], duped_dst_nodes[swap_mask] = duped_dst_nodes[swap_mask], duped_src_nodes[swap_mask]
                src_nodes, dst_nodes = duped_src_nodes, duped_dst_nodes
                edges_to_insert = torch.stack((src_nodes,dst_nodes))
                
                # insertion_indices = 2 * torch.div(orig_insertion_indices, 2, rounding_mode='floor')
                duped_insertion_indices = orig_insertion_indices.repeat_interleave(2)
                incr_mask = torch.arange(start=1,end=reduced_permute_num*2, step=2)
                duped_insertion_indices[incr_mask] += 1
                insertion_indices = duped_insertion_indices
            
            orig_edge_index[:,insertion_indices] = edges_to_insert # modified in place
            # new_edge_index = orig_edge_index.clone().detach()
            # new_edge_index[:,insertion_indices] = edges_to_insert

            # uniques, new_to_uniques_idx, counts = new_edge_index.unique(return_counts=True, return_inverse=True,dim=1)
            # dupes = uniques[:,counts > 1].size()[1]
            # if dupes > 0: breakpoint()

        else:
            # augmentation silently does nothing
            pass

        node_indices = move_to(torch.arange(node_num), self.device)
        node_idx_in_edges = orig_edge_index.flatten()
        combined = torch.cat((node_indices,node_idx_in_edges))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1].sort().values
        intersection = uniques[counts > 1]

        if difference.size()[0] >= 1:
            for i, node_idx in enumerate(difference):
                orig_edge_index = orig_edge_index + (-1 * (orig_edge_index > node_idx)) # - (1 or 0)
            data.edge_index = orig_edge_index
            data.x = data.x[intersection]
            data.num_nodes = intersection.size()[0]
        
        # breakpoint()

        return data

    def extract_subgraph(self, data):
        """
        TODO Would be nice to have the 3rd, as they all model distinct
        ways of conceptualizing types of graph 'semantemes'

        paper uses a random walk to generate a subgraph of nodes
        """
        raise NotImplementedError

    def gsimclr_loss(self, z1, z2):
        # Temporarily put here in framework to minimize sprawl of changes
        # z1 and z2 are the projector embeddings for a pair of graphs
        # See Appendix A Algo 1 https://yyou1996.github.io/files/neurips2020_graphcl_supplement.pdf
        # Code from: https://github.com/Shen-Lab/GraphCL/blob/d857849d51bb168568267e07007c0b0c8bb6d869/transferLearning_MoleculeNet_PPI/bio/pretrain_graphcl.py#L57
        T = 0.1
        batch_size, _ = z1.size()
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', z1_abs, z2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def update(self, batch):

        #######################################################################
        # NOTE How the augmentations are applied in GCL paper codebase...
        #######################################################################
        # 1) In one version of the pretraining reference code, 
        # augmentations are applied in the get(idx) method of 
        # the PyG InMemoryDataset subclass being used. This then always returns
        # the pair of data objects, data, data_aug 
        # See: https://github.com/Shen-Lab/GraphCL/blob/e9e598d478d4a4bff94a3e95a078569c028f1d88/unsupervised_TU/aug.py#L203
        # 2) In a second version, there are 2 dataloaders created and zipped
        # and since they are not shuffled, the orig and augmented graphs are aligned
        # See: https://github.com/Shen-Lab/GraphCL/blob/d857849d51bb168568267e07007c0b0c8bb6d869/transferLearning_MoleculeNet_PPI/chem/pretrain_graphcl.py#L80
        # Thus ... with a two argument loss fn gsimclr_loss(z1, z2), and nice
        # formulation using einsum and subtracting of the diagonal to separate out
        # positive pairs from negative pairs, you can compute the contrastive loss

        # Currently we have the node dropping and edge permuting augmentations
        # implemented, with intention to add subgraph if time allows/useful.
        # See the permute edges function for commentary on the way the original
        # edge permutation was implemented.

        # For shortest path to implementation in this framework as it stands,
        # we're going to start by just applying the augmentation here in line, 
        # after the dataloader has returned a batch of graphs, however
        # initial intuition is that this might be pretty inefficient since it can't
        # be parallelized by the dataloader workers ... instead it's blocking 
        # during training, though, the computations are all torch or native python
        #######################################################################

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
        augmentations = {
            'node_drop': self.drop_nodes,
            'edge_perm': self.permute_edges
        }
        aug_list = list(augmentations.values())

        if self.aug_type ==  'random':
            def rand_aug(data):
                n = torch.randint(2,(1,)).item()
                fn = aug_list[n]
                return fn(data)
            aug_fn = rand_aug
        else:
            if self.aug_type in augmentations:
                aug_fn = augmentations[self.aug_type]
            else:
                raise NotImplementedError

        
        batch_size = x.num_graphs
        # torch.multinomial is slow so we want to do this once per batch
        aug_mask = torch.multinomial(torch.tensor((1-self.aug_prob, self.aug_prob)), batch_size, replacement=True)

        # if not "use_cl" unpack batch of graphs and apply augmentation 
        # to each graph, in place
        # TODO, if "use_cl" create orig, aug pairs,
        # extract features,projections,outputs
        # h1, h2, z1, z2, y_pred1, y_pred2
        # then modify opjective to call new gsimclr loss

        graphs = x.to_data_list()
        
        for i in range(batch_size): 
            if aug_mask[i]:
                graphs[i] = aug_fn(graphs[i])
            else:
                #print("UNCHANGED!")
                pass # original graph kept
        
        x = Batch.from_data_list(graphs)
        
        # Continue as with other methods/ERM
        outputs = self.model(x)
        results['y_pred'] = outputs

        objective = self.objective(results)
        results['objective'] = objective.item()

        # placeholder
        # gsimclr_loss = 0.
        # if isinstance(gsimclr_loss, torch.Tensor):
        #     results['gsimclr_loss'] = gsimclr_loss.item()
        # else:
        #     results['gsimclr_loss'] = gsimclr_loss

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
        sanitized =  self.sanitize_dict(results)
        return sanitized

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
