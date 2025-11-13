from typing import Any

from d2l.base.evaluator import evaluation_metric
from d2l.base.model import Model


@evaluation_metric
def evaluate_rnn_loss(model: Model, test_data_loader: Any) -> float:
    total_loss = 0.0
    total_samples = 0
    for X, y in test_data_loader:
        outputs = model(X)
        y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
        batch_loss = model.loss(y_hat, y)
        batch_size = y.shape[0]
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Test dataloader produced zero samples.")
    return total_loss / total_samples
