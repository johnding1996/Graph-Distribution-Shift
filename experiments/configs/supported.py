from gds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy, MSE, multiclass_logits_to_pred, \
    binary_logits_to_pred, MultiTaskAveragePrecision

from experiments.configs.model import model_defaults
from experiments.configs.algorithm import algorithm_defaults


# algo_log_metrics = {
#     'accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
#     'mse': MSE(),
#     'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
#     'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
#     'multitask_avgprec': MultiTaskAveragePrecision(prediction_fn=None),
#     None: None,
# }

algo_log_metrics = {
    'binary_accuracy': Accuracy(prediction_fn=binary_logits_to_pred),
    'multiclass_accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    'multitask_avgprec': MultiTaskAveragePrecision(prediction_fn=None),
    None: None,
}

process_outputs_functions = {
    'binary_logits_to_pred': binary_logits_to_pred,
    'multiclass_logits_to_pred': multiclass_logits_to_pred,
    None: None,
}

# See models/initializer.py
models = ['gin', 'gcn', 'gin_virtual', 'gcn_virtual']

# See algorithms/initializer.py
algorithms = list(algorithm_defaults.keys())

# See optimizer.py
optimizers = ['SGD', 'Adam', 'AdamW']

# See scheduler.py
schedulers = ['linear_schedule_with_warmup', 'cosine_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR',
              'MultiStepLR']

# See losses.py
losses = ['cross_entropy', 'lm_cross_entropy', 'MSE', 'multitask_bce', 'fasterrcnn_criterion', 'BCEWithLogitsLoss']
