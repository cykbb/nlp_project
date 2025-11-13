from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Generator, List, Tuple, TypedDict, Type

import torch
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from d2l.model import Model
from config import DATASET_CONFIG, METRICS_FILENAME, MODEL_VARIANTS, TRAINING_CONFIG
from dataset import AGNewsDataset
from evaluator import (
    evaluate_text_classification_accuracy,
    evaluate_text_classification_loss,
)
from model import (
    TextClassificationGRU,
    TextClassificationLSTM,
    TextClassificationRNN,
)
from trainer import train_text_classifier

ModelClass = Type[Model]
MODEL_CLASS_REGISTRY: Dict[str, ModelClass] = {
    "lstm": TextClassificationLSTM,
    "gru": TextClassificationGRU,
    "rnn": TextClassificationRNN,
}

class ExperimentResult(TypedDict):
    name: str
    loss: float
    accuracy: float
    checkpoint: str

class EpochMetric(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float

def iter_train_dataloaders(
    dataset: AGNewsDataset, *, batch_size: int, num_epochs: int
) -> Generator[torch.utils.data.DataLoader, None, None]:
    for _ in range(num_epochs):
        yield dataset.get_train_dataloader(
            batch_size=batch_size,
            tokenized=True,
            shuffle=True,
        )


def make_epoch_end_logger(
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    history: List[EpochMetric],
):
    def _callback(model: Model, epoch_id: int, batch_losses: List[float]):
        avg_loss = sum(batch_losses) / max(len(batch_losses), 1)
        val_loss = evaluate_text_classification_loss(model, test_loader)
        val_acc = evaluate_text_classification_accuracy(model, test_loader)
        history.append(
            EpochMetric(
                epoch=epoch_id + 1,
                train_loss=avg_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )
        print(
            f"[{model_name.upper()}][Epoch {epoch_id + 1}] "
            f"train_loss={avg_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    return _callback


def build_model_for_variant(variant_name: str, dataset: AGNewsDataset) -> Model:
    variant_cfg = MODEL_VARIANTS[variant_name]
    model_cls = MODEL_CLASS_REGISTRY[variant_cfg["architecture"]]
    hyperparams = {**variant_cfg["hyperparams"]}
    model_kwargs = {
        **hyperparams,
        "vocab_size": dataset.tokenizer.vocab_size,
        "num_classes": len(dataset.text_labels),
        "pad_token_id": dataset.tokenizer.pad_token_id or 0,
    }
    if variant_cfg["architecture"] == "rnn":
        model_kwargs["nonlinearity"] = variant_cfg.get("nonlinearity", "tanh")
    model = model_cls(**model_kwargs)  # type: ignore[arg-type]
    return model


def train_and_evaluate_variant(
    variant_name: str,
    dataset: AGNewsDataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[ExperimentResult, List[EpochMetric]]:
    print(f"\n=== Training {variant_name.upper()} model ===")
    model = build_model_for_variant(variant_name, dataset)
    checkpoint_name = MODEL_VARIANTS[variant_name]["checkpoint"]
    checkpoint_path = Path(__file__).with_name(checkpoint_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = TRAINING_CONFIG["batch_size"]
    num_epochs = TRAINING_CONFIG["num_epochs"]
    train_loader_epochs = iter_train_dataloaders(
        dataset, batch_size=batch_size, num_epochs=num_epochs
    )
    optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    history: List[EpochMetric] = []

    train_text_classifier(
        model,
        optimizer,
        train_loader_epochs,
        device=device,
        on_epoch_end=make_epoch_end_logger(variant_name, test_loader, history),
        show_progress=True,
    )

    final_loss = evaluate_text_classification_loss(model, test_loader)
    final_acc = evaluate_text_classification_accuracy(model, test_loader)
    model.save(str(checkpoint_path))
    print(
        f"[{variant_name.upper()}] Final evaluation -> "
        f"loss: {final_loss:.4f}, accuracy: {final_acc:.4f}"
    )

    return (
        ExperimentResult(
            name=variant_name,
            loss=final_loss,
            accuracy=final_acc,
            checkpoint=str(checkpoint_path),
        ),
        history,
    )


def main() -> None:
    torch.manual_seed(TRAINING_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AGNewsDataset(**DATASET_CONFIG)
    test_loader = dataset.get_test_dataloader(
        batch_size=TRAINING_CONFIG["batch_size"],
        tokenized=True,
        shuffle=False,
    )

    results: List[ExperimentResult] = []
    metrics: Dict[str, List[EpochMetric]] = {}
    for variant_name in MODEL_VARIANTS:
        result, history = train_and_evaluate_variant(
            variant_name, dataset, test_loader, device
        )
        results.append(result)
        metrics[variant_name] = history

    print("\n=== Model Comparison Summary ===")
    for result in results:
        print(
            f"{result['name'].upper():>5} | "
            f"loss: {result['loss']:.4f} | "
            f"acc: {result['accuracy']:.4f} | "
            f"ckpt: {result['checkpoint']}"
        )

    metrics_path = Path(__file__).with_name(METRICS_FILENAME)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved per-epoch metrics to {metrics_path}")


if __name__ == "__main__":
    main()
