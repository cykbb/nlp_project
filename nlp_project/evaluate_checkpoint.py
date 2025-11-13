from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Type

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from d2l.model import Model
from config import DATASET_CONFIG, MODEL_VARIANTS, TRAINING_CONFIG
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

ModelClass = Type[Model]
MODEL_CLASS_REGISTRY: Dict[str, ModelClass] = {
    "lstm": TextClassificationLSTM,
    "gru": TextClassificationGRU,
    "rnn": TextClassificationRNN,
}

def build_model(dataset: AGNewsDataset, variant_name: str) -> Model:
    variant_cfg = MODEL_VARIANTS[variant_name]
    model_cls = MODEL_CLASS_REGISTRY[variant_cfg["architecture"]]
    kwargs = {
        **variant_cfg["hyperparams"],
        "vocab_size": dataset.tokenizer.vocab_size,
        "num_classes": len(dataset.text_labels),
        "pad_token_id": dataset.tokenizer.pad_token_id or 0,
    }
    if variant_cfg["architecture"] == "rnn":
        kwargs["nonlinearity"] = variant_cfg.get("nonlinearity", "tanh")
    model = model_cls(**kwargs)  # type: ignore[arg-type]
    return model

def load_model_weights(
    model: Model,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints for multiple sequence models."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_VARIANTS.keys()),
        default=None,
        help="Model variants to evaluate (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch size for the evaluation dataloader.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    selected_models: Sequence[str]
    if args.models:
        selected_models = args.models
    else:
        selected_models = list(MODEL_VARIANTS.keys())

    if not selected_models:
        print("No model variants selected for evaluation.")
        return

    torch.manual_seed(TRAINING_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AGNewsDataset(**DATASET_CONFIG)
    test_loader = dataset.get_test_dataloader(
        batch_size=args.batch_size,
        tokenized=True,
        shuffle=False,
    )

    results: List[Dict[str, float | str]] = []
    for model_name in selected_models:
        variant_cfg = MODEL_VARIANTS[model_name]
        checkpoint_path = Path(variant_cfg["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for {model_name} not found: {checkpoint_path}"
            )

        model = build_model(dataset, model_name)
        load_model_weights(model, checkpoint_path, device)
        test_loss = evaluate_text_classification_loss(model, test_loader)
        test_acc = evaluate_text_classification_accuracy(model, test_loader)
        print(
            f"[{model_name.upper()}] loss: {test_loss:.4f}, accuracy: {test_acc:.4f} "
            f"(checkpoint: {checkpoint_path})"
        )
        results.append(
            {
                "name": model_name,
                "loss": test_loss,
                "accuracy": test_acc,
                "checkpoint": str(checkpoint_path),
            }
        )

    if len(results) > 1:
        print("\n=== Evaluation Summary ===")
        for result in results:
            print(
                f"{result['name'].upper():>5} | " # type: ignore
                f"loss: {result['loss']:.4f} | "
                f"acc: {result['accuracy']:.4f} | "
                f"ckpt: {result['checkpoint']}"
            )


if __name__ == "__main__":
    main()
