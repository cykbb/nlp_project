from typing import Any, Callable, Iterable, List, Optional

import torch

from d2l.base.trainer import BatchProcessor, train as base_train
from d2l.base.model import Model
from torch.optim import Optimizer


def _make_rnn_batch_processor(max_grad_norm: Optional[float]) -> BatchProcessor:
    def _processor(model, optimizer, batch):
        X, y = batch
        outputs = model(X)
        y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
        optimizer.zero_grad()
        loss = model.loss(y_hat, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        return float(loss.item())

    return _processor


def train_rnn(
    model: Model,
    optimizer: Optimizer,
    train_data_loaders: Iterable[Any],
    *,
    max_grad_norm: Optional[float] = 1.0,
    on_epoch_end: Optional[Callable[[Model, int, List[float]], Any]] = None,
    show_progress: bool = True,
    batch_processor: Optional[BatchProcessor] = None,
) -> List[List[float]]:
    processor = batch_processor or _make_rnn_batch_processor(max_grad_norm)
    return base_train(
        model=model,
        optimizer=optimizer,
        train_data_loaders=train_data_loaders,
        on_epoch_end=on_epoch_end,
        show_progress=show_progress,
        batch_processor=processor,
    )

