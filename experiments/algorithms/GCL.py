from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from optimizer import initialize_optimizer
from scheduler import initialize_scheduler
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch.nn.utils import clip_grad_norm_
from utils import move_to

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
        # set model components
        self.featurizer = featurizer
        self.projector = projector
        self.classifier = classifier
        # self.model = model #set by base class

        # algorithm hyperparameters
        # save copy in case
        self.config = config
        # If gcl_contrastive_pretrain=False and aug_prob and/or aug_ratio is 0.0, should be equiv to ERM
        self.gcl_contrastive_pretrain = config.gcl_contrastive_pretrain
        self.gcl_reinit_optim_sched = config.gcl_reinit_optim_sched
        self.aug_prob = config.gcl_aug_prob
        # self.aug_type = config.aug_type
        self.aug_type = config.gcl_aug_type
        self.aug_ratio = config.gcl_aug_ratio

        self.gcl_contrast_type = config.gcl_contrast_type
        self.gcl_pretrain_fraction = config.gcl_pretrain_fraction
        self.gcl_finetune_lr_scale =  config.gcl_finetune_lr_scale

        # derived parameters to control switch to finetune
        self.transition_epoch = round(config.n_epochs * self.gcl_pretrain_fraction)
        self.steps_per_epoch = n_train_steps // config.n_epochs
        self.transition_step = self.transition_epoch * self.steps_per_epoch
        self.total_train_steps = n_train_steps
        self.train_step_counter = -1
        self.epoch_counter = -1
        self.is_pretraining = self.gcl_contrastive_pretrain
        self.dummy_output = None

        # sanity check the pretraining augmentation spec
        if self.gcl_contrastive_pretrain==True:
            aug_compatibility = (self.gcl_contrast_type=='orig') or \
                                (self.gcl_contrast_type=='opposite' and self.aug_type != 'random') or \
                                (self.gcl_contrast_type=='random' and self.aug_type == 'random')
            assert aug_compatibility

        # Construct augmentation options
        self.augmentations = {
            'node_drop': self.drop_nodes,
            'edge_perm': self.permute_edges
        }
        self.aug_list = list(self.augmentations.values())

        def rand_aug(data):
            n = torch.randint(2,(1,)).item()
            fn = self.aug_list[n]
            return fn(data)

        # Set main augmentation, also the one applied if doing ERM+Aug
        if self.aug_type ==  'random':
            self.aug_fn = rand_aug
        else:
            if self.aug_type in self.augmentations:
                self.aug_fn = self.augmentations[self.aug_type]
            else:
                raise NotImplementedError
        
        # Set contrastive augmentation relative to main augmentation
        if self.gcl_contrastive_pretrain==True:   
            if self.gcl_contrast_type == 'orig':
                self.orig_aug_fn = lambda data: None
            elif self.gcl_contrast_type == 'random':
                self.orig_aug_fn = rand_aug
            elif self.gcl_contrast_type == 'opposite':
                if self.aug_type=='node_drop':
                    self.orig_aug_fn = self.augmentations['edge_perm']  
                else:
                    self.orig_aug_fn = self.augmentations['node_drop']
            else:
                raise NotImplementedError

        # The logging stuff always gives me errors honestly
        # if self.gcl_contrastive_pretrain==True:
            # self.logged_fields.append('train_step_counter')
            # self.logged_fields.append('epoch_counter')


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

        #######################################################################
        # prototyping degree weighted drop
        weighted_drop = True
        if weighted_drop and drop_num > 0:
            out_degrees = degree(orig_edge_index[0],num_nodes=node_num,dtype=int)
            if data.is_directed():
                in_degrees = degree(orig_edge_index[1],num_nodes=node_num,dtype=int)
                io_degrees = out_degrees + in_degrees
            else:
                io_degrees = out_degrees

            if data.contains_isolated_nodes():
                # not sure how principled this is?
                io_degrees[io_degrees==0] = 0.1
            n_power = 4.0  #2.0
            io_degrees = io_degrees ** (n_power)

            deg_probs = io_degrees / io_degrees.sum()
            drop_probs = 1. - deg_probs
            # the orig code seems to drop weight the "highest degree" rather 
            # than the inverse? but commented code suggests they were trying both maybe?
            # can't figure out what power they used, "n" could stand for negative? 
            idx_to_drop = torch.multinomial(drop_probs, drop_num, replacement=False)
            idx_to_not_drop = torch.tensor([n for n in range(node_num) if not n in idx_to_drop], device = orig_edge_index.device)

            idx_nondrop = idx_to_not_drop

            # breakpoint()
        #######################################################################

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
        except:
            pass
            # data = data

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
        node_num = data.x.size()[0]
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

            orig_edge_index[:,insertion_indices] = edges_to_insert # modified in place
        else:
            # augmentation silently does nothing
            pass


    def extract_subgraph(self, data):
        """
        TODO Would be nice to have the 3rd, as they all model distinct
        ways of conceptualizing types of graph 'semantemes'

        paper uses a random walk to generate a subgraph of nodes
        """
        raise NotImplementedError

    def gsimclr_loss(self, z1, z2):
        # Temporarily put here in framework to minimize sprawl of changes
        # z1 and z2 are the projector embeddings for two batches of graphs (batch_size, proj_embedding)
        # See Appendix A Algo 1 https://yyou1996.github.io/files/neurips2020_graphcl_supplement.pdf
        # Code from: https://github.com/Shen-Lab/GraphCL in file transferLearning_MoleculeNet_PPI/bio/pretrain_graphcl.py
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
        """
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

        # Currently we have the node dropping and edge permuting augmentations
        # implemented, with intention to add subgraph if time allows/useful.
        # See the permute edges function for commentary on the way the original
        # edge permutation was implemented.

        # The shortest path to implementation in this framework was
        # just applying the augmentation here in line, 
        # after the dataloader has returned a batch of graphs, however
        # initial intuition is that this might be pretty inefficient since it can't
        # be parallelized by the dataloader workers ... instead it's blocking 
        # during training, though, the computations are all torch or native python
        #######################################################################
        """
        # Block to count steps/epochs until time to switch to finetuning/ERM
        # when finished with contrastive pretrain phase
        if self.gcl_contrastive_pretrain==True:
            self.train_step_counter += 1
            if self.train_step_counter % self.steps_per_epoch == 0:
                self.epoch_counter += 1
            if self.is_pretraining == True:
                if self.train_step_counter >= self.transition_step:
                    self.is_pretraining = False 
                    print("Switching from self-supervised GCL pretraining to standard ERM (no augmentation) finetuning")
                    self.aug_prob = 0.0  
                    # ^ this makes the multinomial draw for the augmentation mask degenerate to [0, 0, 0 ...]

                    if self.config.gcl_reinit_optim_sched:
                        # modify the lr, as a fraction of the original, maybe 1.0 though
                        self.config.lr = self.config.lr*self.config.gcl_finetune_lr_scale
                        # reset the optimizer, and init the schedule based on remaining train steps
                        self.optimizer = initialize_optimizer(self.config, self.model)
                        scheduler = initialize_scheduler(self.config, self.optimizer, (self.total_train_steps-self.train_step_counter))
                        self.schedulers=[scheduler, ]

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

        # Create the masks used to implement the randomness of the "aug_prob" parameter
        batch_size = x.num_graphs
        aug_mask = torch.multinomial(torch.tensor((1-self.aug_prob, self.aug_prob)), batch_size, replacement=True)
        if self.is_pretraining:
            orig_aug_mask = torch.multinomial(torch.tensor((1-self.aug_prob, self.aug_prob)), batch_size, replacement=True)

        # When performing "gcl_contrastive_pretrain" unpack batch of graphs 
        # and create a copy of them, before applying the respective augmentation type to each.
        # If just doing ERM+Aug, skip the copying and pair wise operations. 
        # Forgive the fact that "orig" is often operated on second, implementation artifact
        
        graphs = x.to_data_list()

        if self.is_pretraining:
            orig_graphs = [x.clone() for x in graphs]

        for i in range(batch_size):
            if aug_mask[i]:
                self.aug_fn(graphs[i])
            if self.is_pretraining and orig_aug_mask[i]:
                self.orig_aug_fn(orig_graphs[i])
        
        x = Batch.from_data_list(graphs)
        if self.is_pretraining:
            orig_x = Batch.from_data_list(orig_graphs)

        # The loss calculation for "gcl_contrastive_pretrain" involves feeding the two
        # lists of pairwise augmented graphs through the featurizer and projector
        # before the simclr style loss. 
        if self.is_pretraining:
            # TODO ignore this its just to prevent logging error
            # do once per training run
            if self.dummy_output == None:
                self.dummy_output = self.model(x)
            results['y_pred'] = self.dummy_output[:batch_size] 

            z_0 = self.projector(self.featurizer(orig_x))
            z_1 = self.projector(self.featurizer(x))
            results['z_0'], results['z_1'] = z_0, z_1
            # Just directly calling the loss here not using the objective() interface
            objective = self.gsimclr_loss(z_0, z_1)
            results['objective'] = objective.item()

        else: 
            # Continue as with other methods/ERM
            # this block runs when just doing ERM+Aug as well
            outputs = self.model(x)
            results['y_pred'] = outputs

            objective = self.objective(results)
            results['objective'] = objective.item()
        
        # Regardless of which loss used, do the backwards pass the same way
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
