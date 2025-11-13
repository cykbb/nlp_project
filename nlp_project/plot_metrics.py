from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import METRICS_FILENAME

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training curves (train loss, val loss, val accuracy) from metrics JSON."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(__file__).with_name(METRICS_FILENAME),
        help="Path to the JSON file produced by main training script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> Dict[str, List[Dict[str, float]]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_metric(
    metrics: Dict[str, List[Dict[str, float]]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    show: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, history in metrics.items():
        if not history:
            continue
        epochs = [entry["epoch"] for entry in history]
        values = [entry[metric_key] for entry in history]
        plt.plot(epochs, values, marker="o", label=model_name.upper())

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)

    plot_metric(
        metrics,
        metric_key="train_loss",
        ylabel="Train Loss",
        output_path=args.output_dir / "train_loss.png",
        show=args.show,
    )
    plot_metric(
        metrics,
        metric_key="val_loss",
        ylabel="Validation/Test Loss",
        output_path=args.output_dir / "val_loss.png",
        show=args.show,
    )
    plot_metric(
        metrics,
        metric_key="val_accuracy",
        ylabel="Validation/Test Accuracy",
        output_path=args.output_dir / "val_accuracy.png",
        show=args.show,
    )
    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
