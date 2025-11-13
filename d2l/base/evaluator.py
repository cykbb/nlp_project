from functools import wraps
from typing import Any, Callable, TypeVar, cast

import torch

from d2l.base.model import Model

MetricFunc = Callable[[Model, Any], Any]
MetricFuncT = TypeVar("MetricFuncT", bound=MetricFunc)


def evaluation_metric(metric_fn: MetricFuncT) -> MetricFuncT:
    @wraps(metric_fn)
    def wrapper(model: Model, test_data_loader: Any) -> float:
        model.eval()
        with torch.no_grad():
            return metric_fn(model, test_data_loader)

    return cast(MetricFuncT, wrapper)


@evaluation_metric
def evaluate_loss(model: Model, test_data_loader: Any) -> float:
    total_loss = 0.0
    total_samples = 0
    for X, y in test_data_loader:
        y_hat = model(X)
        batch_loss = model.loss(y_hat, y)
        batch_size = y.shape[0]
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Test dataloader produced zero samples.")
    return total_loss / total_samples


@evaluation_metric
def evaluate_accuracy(model: Model, test_data_loader: Any) -> float:
    correct, total = 0, 0
    for X, y in test_data_loader:
        y_hat = model(X)
        correct += (y_hat.argmax(dim=1) == y).sum().item()
        total += y.shape[0]
    if total == 0:
        raise ValueError("Test dataloader produced zero samples.")
    return correct / total
