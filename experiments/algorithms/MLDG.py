import torch
import torch.nn.functional as F
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to
import copy
from gds.common.utils import split_into_groups
from torch_geometric.data import Batch

class MLDG(SingleModelAlgorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    # def __init__(self, input_shape, num_classes, num_domains,
    #              hparams, conditional, class_balance):
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps):
        model = initialize_model(config, d_out=d_out, is_featurizer=False).to(config.device)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.config = config


    def update(self, batch):

        x, y_true, metadata = batch

        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata,
        }

        self.optimizer.zero_grad()
        for p in self.model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        unique_groups, group_indices, _ = split_into_groups(results['g'])
        n_groups_per_batch = unique_groups.numel()
        rand_group_index = torch.randperm(n_groups_per_batch)
        group_pairs = []
        for i in range(rand_group_index.shape[0]) :
            j = i + 1 if i < rand_group_index.shape[0]-1 else 0
            group_i, group_j = rand_group_index[i], rand_group_index[j]
            group_pairs.append((group_indices[group_i], group_indices[group_j]))

        objective = 0
        for (group_indices_i, group_indices_j) in group_pairs:
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.model)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.config.lr,
                weight_decay=0
            )

            inner_obj = self.objective({
                'y_pred': inner_net(Batch.from_data_list(x[group_indices_i])),
                'y_true': y_true[group_indices_i]
            })

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.model.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / n_groups_per_batch)

            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = self.objective({
                'y_pred': inner_net(Batch.from_data_list(x[group_indices_j])),
                'y_true': y_true[group_indices_j]
            })
            grad_inner_j = torch.autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)
            # `objective` is populated for reporting purposes
            hparams_mldg_beta = 1
            objective += (hparams_mldg_beta * loss_inner_j).item()


            for p, g_j in zip(self.model.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(hparams_mldg_beta * g_j.data / n_groups_per_batch)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        self.optimizer.step()
        objective /= n_groups_per_batch

        results['objective'] = objective
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)

        # log results

        results['y_pred'] = self.model(x)
        self.update_log(results)
        return self.sanitize_dict(results)

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
