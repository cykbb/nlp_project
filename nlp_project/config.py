"""Shared hyperparameters and file names for the NLP project."""

from __future__ import annotations

from typing import Dict, Literal, TypedDict

class DatasetConfig(TypedDict):
    tokenizer_name: str
    max_length: int

class ModelConfig(TypedDict):
    embedding_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool

class TrainingConfig(TypedDict):
    batch_size: int
    num_epochs: int
    learning_rate: float
    seed: int

class _ModelVariantRequired(TypedDict):
    hyperparams: ModelConfig
    checkpoint: str
    architecture: Literal["lstm", "gru", "rnn"]

class ModelVariantConfig(_ModelVariantRequired, total=False):
    nonlinearity: str

METRICS_FILENAME = "training_metrics.json"

DATASET_CONFIG: DatasetConfig = {
    "tokenizer_name": "bert-base-uncased",
    "max_length": 128,
}

TRAINING_CONFIG: TrainingConfig = {
    "batch_size": 64,
    "num_epochs": 5,
    "learning_rate": 1e-3,
    "seed": 42,
}

def _default_model_hyperparams() -> ModelConfig:
    return {
        "embedding_dim": 128,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
    }

MODEL_VARIANTS: Dict[str, ModelVariantConfig] = {
    "lstm": {
        "architecture": "lstm",
        "checkpoint": "text_lstm_agnews.pt",
        "hyperparams": _default_model_hyperparams(),
    },
    "gru": {
        "architecture": "gru",
        "checkpoint": "text_gru_agnews.pt",
        "hyperparams": _default_model_hyperparams(),
    },
    "rnn": {
        "architecture": "rnn",
        "checkpoint": "text_rnn_agnews.pt",
        "hyperparams": _default_model_hyperparams(),
        "nonlinearity": "tanh",
    },
}
