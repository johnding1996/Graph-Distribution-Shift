import torch
import torch.nn.functional as F
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to

class AbstractDANN(SingleModelAlgorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    # def __init__(self, input_shape, num_classes, num_domains,
    #              hparams, conditional, class_balance):
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps, conditional, class_balance):
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        model = torch.nn.Sequential(featurizer, classifier).to(config.device)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        assert config.num_domains <= 1000 # domain space shouldn't be too large

        self.featurizer = featurizer
        self.classifier = classifier
        self.register_buffer('update_count', torch.tensor([0]))


##############################################
        self.conditional = conditional
        self.class_balance = class_balance
        num_classes = d_out
        emb_dim = self.featurizer.d_out
        self.discriminator = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                  torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                  torch.nn.Linear(emb_dim, config.num_domains)).to(config.device)
        self.class_embeddings = torch.nn.Embedding(num_classes,
            self.featurizer.d_out).to(config.device)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=config.lr,
            weight_decay=0,
            betas=(0.5, 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=config.lr,
            weight_decay=0,
            betas=(0.5, 0.9))

    def process_batch(self, batch):
        """
        Override
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        features = self.featurizer(x)
        outputs = self.classifier(features)

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'features': features,
        }
        return results

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

        self.update_count += 1
        z = self.featurizer(x)
        if self.conditional:
            disc_input = z + self.class_embeddings(y_true)
        else:
            disc_input = z
        disc_out = self.discriminator(disc_input)

        # should be the domain label
        disc_labels = move_to(metadata[:,0].flatten(), self.device)

        if self.class_balance:
            y_counts = F.one_hot(y_true).sum(dim=0)
            weights = 1. / (y_counts[y_true] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = torch.autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        hparams_gra_penalty = 0
        disc_loss += hparams_gra_penalty * grad_penalty

        d_steps_per_g = hparam_d_steps_per_g_step = 1
        all_preds = self.classifier(z)
        results['y_pred'] = all_preds
        classifier_loss = self.objective(results)

        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
        else:
            hparams_lambda = 1.0
            gen_loss = (classifier_loss +
                        (hparams_lambda * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()

        results['objective'] = classifier_loss.item()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)

        # log results
        self.update_log(results)
        return self.sanitize_dict(results)

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps):
        super(DANN, self).__init__(config, d_out, grouper, loss,
                 metric, n_train_steps, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps):
        super(CDANN, self).__init__(config, d_out, grouper, loss,
                 metric, n_train_steps, conditional=True, class_balance=True)