from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model


class GSN(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
                 metric, n_train_steps, full_dataset=None):
        model = initialize_model(config, d_out, full_dataset=full_dataset).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
