from __future__ import annotations

from typing import Iterable, Mapping, Optional, cast

import torch
from torch.optim import Optimizer
from torch import Tensor

from d2l.base import trainer as base_trainer
from d2l.base.model import Model
from d2l.base.trainer import BatchProcessor, EpochCallback

BatchMapping = Mapping[str, Tensor]

def _prepare_batch(
    batch: BatchMapping, device: torch.device
) -> tuple[Tensor, Optional[Tensor], Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch["labels"].to(device)
    return input_ids, attention_mask, labels


def make_text_batch_processor(
    *,
    device: torch.device,
    max_grad_norm: Optional[float] = 1.0,
) -> BatchProcessor:
    def processor(
        model: Model,
        optimizer: Optimizer,
        batch: BatchMapping,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        input_ids, attention_mask, labels = _prepare_batch(batch, device)
        logits = model(input_ids, attention_mask=attention_mask)
        loss = model.loss(logits, labels)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        return float(loss.item())

    return processor


def train_text_classifier(
    model: Model,
    optimizer: Optimizer,
    train_data_loaders: Iterable[Iterable[BatchMapping]],
    *,
    device: Optional[torch.device] = None,
    max_grad_norm: Optional[float] = 1.0,
    on_epoch_end: Optional[EpochCallback] = None,
    show_progress: bool = True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cast(Model, model.to(device))
    batch_processor = make_text_batch_processor(
        device=device,
        max_grad_norm=max_grad_norm,
    )
    return base_trainer.train(
        model,
        optimizer,
        train_data_loaders,
        on_epoch_end=on_epoch_end,
        show_progress=show_progress,
        batch_processor=batch_processor,
    )
