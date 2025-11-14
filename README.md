# NLP Text Classification Project

A text classification project based on Recurrent Neural Networks (RNN/LSTM/GRU), using the AG News dataset for news classification tasks.

## Project Architecture

### Directory Structure

```
nlp_project/
├── pyproject.toml          # Project dependency configuration
├── README.md               # Project documentation
├── d2l/                    # Deep learning framework base modules
│   ├── __init__.py
│   ├── dataset.py         # Dataset base class
│   ├── evaluator.py       # Evaluator base class
│   ├── model.py           # Model base class
│   ├── plot.py            # Plotting utilities
│   └── trainer.py         # Trainer base class
└── nlp_project/           # Core implementation code
    ├── config.py          # Configuration file (hyperparameters, model variants)
    ├── dataset.py         # AG News dataset loading and preprocessing
    ├── model.py           # RNN/LSTM/GRU model implementations
    ├── trainer.py         # Text classification trainer
    ├── evaluator.py       # Text classification evaluator
    ├── main.py            # Main training script
    ├── evaluate_checkpoint.py  # Model evaluation script
    ├── plot_metrics.py    # Training metrics visualization
    └── exp1/ & exp2/      # Experimental results directory
```

### Core Module Description

#### 1. **Data Module** (`dataset.py`)
- **AGNewsDataset**: AG News dataset wrapper class
  - Supports HuggingFace `transformers` tokenizer
  - Automatic text cleaning (removes HTML tags and URLs)
  - Provides both raw text and tokenized data formats
  - Dataset contains 4 categories: World, Sports, Business, Sci/Tech
  - Training set: 120,000 samples
  - Test set: 7,600 samples

#### 2. **Model Module** (`model.py`)
- **TextClassificationModel**: Text classification base model class
  - Shared embedding layer and classifier structure
  - Supports bidirectional RNN encoding
  - Uses pack_padded_sequence for optimized variable-length sequence processing
  
- **TextClassificationLSTM**: BiLSTM text classifier
- **TextClassificationGRU**: BiGRU text classifier
- **TextClassificationRNN**: Standard RNN text classifier (supports tanh/relu activation)

#### 3. **Configuration Module** (`config.py`)
- **Dataset Configuration**:
  - Tokenizer: `bert-base-uncased`
  - Max sequence length: 128
  
- **Training Configuration**:
  - Batch size: 64
  - Epochs: 5
  - Learning rate: 1e-3
  - Random seed: 42
  
- **Model Hyperparameters**:
  - Embedding dim: 128
  - Hidden size: 256
  - Num layers: 2
  - Dropout: 0.2
  - Bidirectional: True

#### 4. **Training and Evaluation Modules**
- `trainer.py`: Implements text classifier training loop
- `evaluator.py`: Implements loss and accuracy evaluation
- `main.py`: Main training script with multi-model comparison support

## Usage

### Environment Setup

1. **Install Dependencies**

```bash
# Install using pip
pip install -e .

# Or manually install main dependencies
pip install torch torchvision transformers datasets tqdm matplotlib numpy pandas
```

2. **Python Version Requirement**: >= 3.11

### Training Models

#### Basic Training

Run the main training script to automatically train all model variants (LSTM, GRU, RNN):

```bash
cd nlp_project
python main.py
```

The training process will:
- Automatically download the AG News dataset
- Train LSTM, GRU, and RNN models sequentially
- Output training and validation metrics after each epoch
- Save model checkpoints to the `exp1/` directory
- Save training metrics to `training_metrics.json`

#### Custom Configuration

Modify configuration parameters in `config.py`:

```python
# Modify training configuration
TRAINING_CONFIG = {
    "batch_size": 64,       # Batch size
    "num_epochs": 5,        # Number of epochs
    "learning_rate": 1e-3,  # Learning rate
    "seed": 42,             # Random seed
}

# Modify model hyperparameters
def _default_model_hyperparams():
    return {
        "embedding_dim": 128,    # Embedding dimension
        "hidden_size": 256,      # Hidden layer size
        "num_layers": 2,         # Number of RNN layers
        "dropout": 0.2,          # Dropout probability
        "bidirectional": True,   # Bidirectional or not
    }
```

### Evaluate Models

Evaluate saved model checkpoints:

```bash
python evaluate_checkpoint.py
```

### Visualize Training Process

Plot loss and accuracy curves during training:

```bash
python plot_metrics.py
```

## Experimental Results

### Experiment 1 (exp1)

All models trained with the same hyperparameters (5 epochs):

| Model | Final Test Loss | Final Test Accuracy | Training Time/Epoch |
|------|-------------|---------------|---------------|
| **LSTM** | **0.2379** | **92.47%** | ~15 minutes |
| **GRU** | 0.2421 | 92.43% | ~12 minutes |
| **RNN** | 0.3712 | 88.64% | ~6 minutes |

### Experiment 2 (exp2)

Detailed training process metrics:

#### LSTM Training Curve
| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|-------|---------|---------|-----------|
| 1 | 0.5003 | 0.3006 | 89.71% |
| 2 | 0.2625 | 0.2702 | 91.21% |
| 3 | 0.2032 | 0.2328 | 92.12% |
| 4 | 0.1624 | 0.2296 | 92.25% |
| 5 | 0.1309 | 0.2376 | **92.50%** |

#### GRU Training Curve
| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|-------|---------|---------|-----------|
| 1 | 0.4733 | 0.2852 | 89.79% |
| 2 | 0.2442 | 0.2377 | 91.92% |
| 3 | 0.1856 | 0.2340 | 92.20% |
| 4 | 0.1495 | 0.2330 | 92.34% |
| 5 | 0.1226 | 0.2500 | **92.28%** |

#### RNN Training Curve
| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|-------|---------|---------|-----------|
| 1 | 0.8534 | 0.7698 | 67.36% |
| 2 | 0.5901 | 0.4386 | 84.64% |
| 3 | 0.4838 | 0.4142 | 85.38% |
| 4 | 0.4120 | 0.4812 | 85.22% |
| 5 | 0.3699 | 0.3712 | **88.64%** |

### Results Analysis

1. **Performance Comparison**:
   - LSTM and GRU have similar performance, both achieving **92%+ accuracy**
   - LSTM slightly outperforms GRU (92.47% vs 92.43%)
   - Standard RNN has noticeably worse performance (88.64%), limited by vanishing gradient problem

2. **Training Efficiency**:
   - GRU trains fastest (~12 minutes/epoch), about 20% faster than LSTM
   - RNN trains fastest (~6 minutes/epoch), but with significant performance sacrifice
   - LSTM is slower but achieves best final performance

3. **Convergence Characteristics**:
   - LSTM and GRU converge stably with continuously decreasing loss
   - RNN shows validation loss increase at epoch 4 (signs of overfitting)
   - All models basically converge within 5 epochs

4. **Model Selection Recommendations**:
   - **For highest accuracy**: Choose LSTM
   - **Balance performance and speed**: Choose GRU
   - **Fast prototyping**: Can use RNN

## Main Dependencies

- **PyTorch** >= 2.8.0: Deep learning framework
- **Transformers** >= 4.57.1: HuggingFace tokenizer
- **Datasets** >= 3.2.0: Dataset loading
- **tqdm** >= 4.67.1: Progress bar display
- **matplotlib** >= 3.10.6: Visualization
- **numpy** >= 2.3.3: Numerical computation
- **pandas** >= 2.3.2: Data processing

## Project Features

1. **Modular Design**: Clear code structure, easy to extend and maintain
2. **Multi-Model Support**: Supports three recurrent neural network architectures: RNN, LSTM, GRU
3. **Flexible Configuration**: Unified configuration management for convenient hyperparameter tuning
4. **Complete Training Pipeline**: Includes full workflow of data loading, model training, evaluation, and visualization
5. **High-Performance Implementation**: Uses pack_padded_sequence to optimize variable-length sequence processing
6. **Bidirectional RNN**: Fully utilizes contextual information to improve classification performance

## License

This project is for learning and research purposes only.
