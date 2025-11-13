from typing import Any, Callable, Iterable, List, Optional

from torch.optim import Optimizer
from tqdm.auto import tqdm  # type: ignore

from d2l.base.model import Model

Batch = Any
BatchProcessor = Callable[[Model, Optimizer, Batch], float]
EpochCallback = Callable[[Model, int, List[float]], Any]

def _default_batch_processor(model: Model, optimizer: Optimizer, batch: Batch) -> float:
    X, y = batch
    optimizer.zero_grad()
    y_hat = model(X)
    loss = model.loss(y_hat, y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _maybe_progress(iterable: Iterable[Any], show_progress: bool, **kwargs: Any) -> Iterable[Any]:
    if show_progress:
        return tqdm(iterable, **kwargs)
    return iterable

def train(
    model: Model,
    optimizer: Optimizer,
    train_data_loaders: Iterable[Any],
    *,
    on_epoch_end: Optional[Callable[[Model, int, List[float]], Any]] = None,
    show_progress: bool = True,
    batch_processor: Optional[BatchProcessor] = None,
) -> List[List[float]]:
    processor = batch_processor or _default_batch_processor
    epoch_losses: List[List[float]] = []
    epoch_iter = _maybe_progress(
        enumerate(train_data_loaders),
        show_progress,
        desc="Epochs",
        leave=True,
    )
    for epoch_id, train_data_loader in epoch_iter:
        model.train()
        batch_losses: List[float] = []
        batch_iter = _maybe_progress(
            train_data_loader,
            show_progress,
            desc=f"Epoch {epoch_id}",
            leave=False,
        )
        for batch in batch_iter:
            batch_loss = processor(model, optimizer, batch)
            batch_losses.append(batch_loss)
        epoch_losses.append(batch_losses)
        if on_epoch_end:
            on_epoch_end(model, epoch_id, batch_losses)
    return epoch_losses
