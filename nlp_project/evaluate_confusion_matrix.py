"""
Evaluate trained models and generate confusion matrix with metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DATASET_CONFIG, MODEL_VARIANTS, TRAINING_CONFIG
from dataset import AGNewsDataset
from model import (
    TextClassificationGRU,
    TextClassificationLSTM,
    TextClassificationRNN,
)


def get_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions and true labels.
    
    Returns:
        predictions: numpy array of predicted labels
        true_labels: numpy array of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            
            logits = model(input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: Path,
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix using seaborn heatmap.
    
    Args:
        cm: confusion matrix
        class_names: list of class names
        model_name: name of the model
        save_path: path to save the figure
        normalize: whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'Normalized Confusion Matrix - {model_name.upper()}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name.upper()}'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def evaluate_model(
    model_name: str,
    dataset: AGNewsDataset,
    device: torch.device,
    output_dir: Path
) -> None:
    """
    Evaluate a single model and generate metrics and confusion matrix.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name.upper()} model")
    print(f"{'=' * 60}")
    
    # Load model
    variant_cfg = MODEL_VARIANTS[model_name]
    checkpoint_path = Path(__file__).parent / variant_cfg["checkpoint"]
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Build model
    architecture = variant_cfg["architecture"]
    hyperparams = variant_cfg["hyperparams"]
    
    model_kwargs = {
        **hyperparams,
        "vocab_size": dataset.tokenizer.vocab_size,
        "num_classes": len(dataset.text_labels),
        "pad_token_id": dataset.tokenizer.pad_token_id or 0,
    }
    
    if architecture == "lstm":
        model = TextClassificationLSTM(**model_kwargs)
    elif architecture == "gru":
        model = TextClassificationGRU(**model_kwargs)
    elif architecture == "rnn":
        model_kwargs["nonlinearity"] = variant_cfg.get("nonlinearity", "tanh")
        model = TextClassificationRNN(**model_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load checkpoint
    model.load(str(checkpoint_path))
    model = model.to(device)
    
    # Get test dataloader
    test_loader = dataset.get_test_dataloader(
        batch_size=TRAINING_CONFIG["batch_size"],
        tokenized=True,
        shuffle=False
    )
    
    # Get predictions
    print("Getting predictions...")
    predictions, true_labels = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"\nEvaluation Metrics for {model_name.upper()}:")
    print(f"{'-' * 60}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")
    print(f"{'-' * 60}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'-' * 60}")
    precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
    
    for i, class_name in enumerate(dataset.text_labels):
        print(f"{class_name:12} - Precision: {precision_per_class[i]:.4f}, "
              f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")
    
    # Plot confusion matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot absolute counts
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        cm, 
        dataset.text_labels, 
        model_name, 
        cm_path,
        normalize=False
    )
    
    # Plot normalized version
    cm_norm_path = output_dir / f"{model_name}_confusion_matrix_normalized.png"
    plot_confusion_matrix(
        cm, 
        dataset.text_labels, 
        model_name, 
        cm_norm_path,
        normalize=True
    )
    
    # Save metrics to file
    metrics_path = output_dir / f"{model_name}_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Metrics for {model_name.upper()}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted):    {recall:.4f}\n")
        f.write(f"F1-score (weighted):  {f1:.4f}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'-' * 60}\n")
        for i, class_name in enumerate(dataset.text_labels):
            f.write(f"{class_name:12} - Precision: {precision_per_class[i]:.4f}, "
                   f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\nSaved metrics to {metrics_path}")


def plot_all_models_comparison(
    dataset: AGNewsDataset,
    device: torch.device,
    output_dir: Path
) -> None:
    """
    Plot confusion matrices for all models in a single figure for comparison.
    """
    n_models = len(MODEL_VARIANTS)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(MODEL_VARIANTS.keys()):
        variant_cfg = MODEL_VARIANTS[model_name]
        checkpoint_path = Path(__file__).parent / variant_cfg["checkpoint"]
        
        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: checkpoint not found")
            continue
        
        # Build and load model
        architecture = variant_cfg["architecture"]
        hyperparams = variant_cfg["hyperparams"]
        
        model_kwargs = {
            **hyperparams,
            "vocab_size": dataset.tokenizer.vocab_size,
            "num_classes": len(dataset.text_labels),
            "pad_token_id": dataset.tokenizer.pad_token_id or 0,
        }
        
        if architecture == "lstm":
            model = TextClassificationLSTM(**model_kwargs)
        elif architecture == "gru":
            model = TextClassificationGRU(**model_kwargs)
        elif architecture == "rnn":
            model_kwargs["nonlinearity"] = variant_cfg.get("nonlinearity", "tanh")
            model = TextClassificationRNN(**model_kwargs)
        
        model.load(str(checkpoint_path))
        model = model.to(device)
        
        # Get predictions
        test_loader = dataset.get_test_dataloader(
            batch_size=TRAINING_CONFIG["batch_size"],
            tokenized=True,
            shuffle=False
        )
        predictions, true_labels = get_predictions(model, test_loader, device)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        ax = axes[idx]
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=dataset.text_labels,
            yticklabels=dataset.text_labels,
            ax=ax,
            cbar_kws={'label': 'Proportion'}
        )
        ax.set_title(f'{model_name.upper()}\nAccuracy: {accuracy:.2%}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = output_dir / "all_models_confusion_matrix_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison figure to {comparison_path}")


def main():
    """Main evaluation function."""
    # Setup
    torch.manual_seed(TRAINING_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = AGNewsDataset(**DATASET_CONFIG)
    print(f"Dataset loaded: {dataset.train_size} train, {dataset.test_size} test")
    print(f"Classes: {dataset.text_labels}")
    
    # Output directory - use exp3 for latest results
    output_dir = Path(__file__).parent / "exp3" / "evaluation_metrics"
    
    # Evaluate each model
    for model_name in MODEL_VARIANTS.keys():
        evaluate_model(model_name, dataset, device, output_dir)
    
    # Create comparison plot
    print(f"\n{'=' * 60}")
    print("Creating comparison plot for all models...")
    print(f"{'=' * 60}")
    plot_all_models_comparison(dataset, device, output_dir)
    
    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
