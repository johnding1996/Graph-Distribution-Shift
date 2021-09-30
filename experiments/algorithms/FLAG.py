from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import torch
from torch.nn.utils import clip_grad_norm_
from utils import move_to


class FLAG(SingleModelAlgorithm):
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
        self.config = config

    def update(self, batch):
        step_size = self.config.flag_step_size
        m = 3

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


        perturb_shape = (x.x.shape[0], self.model.emb_dim)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size)
        perturb = move_to(perturb, self.device)
        perturb.requires_grad_()

        self.optimizer.zero_grad()

        outputs = self.model(x, perturb)
        results['y_pred'] = outputs
        objective = self.objective(results)
        objective /= m

        for _ in range(m - 1):
            objective.backward()
            perturb_data = perturb.detach() + step_size*torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            outputs = self.model(x, perturb)
            results['y_pred'] = outputs
            objective = self.objective(results)
            objective /= m

        objective.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        results['objective'] = objective.item() * m
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)

        # log results
        self.update_log(results)
        return self.sanitize_dict(results)

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
