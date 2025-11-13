from __future__ import annotations

from typing import Mapping

import torch

from d2l.evaluator import evaluation_metric
from d2l.model import Model

BatchMapping = Mapping[str, torch.Tensor]

def _prepare_eval_tensors(
    batch: BatchMapping, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    return input_ids, attention_mask, labels

@evaluation_metric
def evaluate_text_classification_loss(
    model: Model,
    data_loader: torch.utils.data.DataLoader,
) -> float:
    device = next(model.parameters()).device
    total_loss = 0.0
    total_samples = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = _prepare_eval_tensors(batch, device)
        logits = model(input_ids, attention_mask=attention_mask)
        batch_loss = model.loss(logits, labels)
        batch_size = labels.size(0)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Test dataloader produced zero samples.")
    return total_loss / total_samples


@evaluation_metric
def evaluate_text_classification_accuracy(
    model: Model,
    data_loader: torch.utils.data.DataLoader,
) -> float:
    device = next(model.parameters()).device
    correct = 0
    total = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = _prepare_eval_tensors(batch, device)
        logits = model(input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("Test dataloader produced zero samples.")
    return correct / total
