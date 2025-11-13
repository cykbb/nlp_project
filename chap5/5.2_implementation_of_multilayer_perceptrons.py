#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[2]:


import importlib
from d2l.classification.model import MLPClassifierTorch, MLPClassifier
from d2l.classification.dataset import FashionMNISTDataset
from d2l.base.trainer import train
from d2l.base.evaluator import evaluate_accuracy, evaluate_loss
from d2l.base.optimizer import SGDOptimizer
from d2l.base.plot import plot_loss, show_images
from d2l.base.utils import mps
from typing import List
from matplotlib import pyplot as plt


# In[3]:


import torch


# In[4]:


rng = torch.Generator().manual_seed(42)


# In[5]:


data = FashionMNISTDataset()
mlp_model_scratch = MLPClassifier(
    num_features=784, 
    num_outputs=10,
    num_hiddens=[256, 256],
    rng=rng
)
mlp_model = MLPClassifierTorch(
    num_features=784, 
    num_outputs=10, 
    num_hiddens=[256, 256],
    rng=rng
)


# In[6]:


# Let's check the shape of the data
train_loader = data.get_train_dataloader(256)
for X, y in train_loader:
    print(f"Input shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    break


# In[7]:


def on_epoch_end(model, epoch: int, losses: List[float]) -> None:
    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    test_loader = data.get_test_dataloader(data.test_size)
    test_acc = evaluate_accuracy(model, test_loader)
    test_loss = evaluate_loss(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


# In[8]:


from d2l import plot_losses

optimizer_scratch = SGDOptimizer(list(mlp_model_scratch.parameters()), lr=0.1)
all_epoch_losses_scratch = train(
    model=mlp_model_scratch,
    optimizer=optimizer_scratch,
    train_data_loaders=data.get_train_dataloader_epochs(256, 10),
    on_epoch_end=on_epoch_end,
)

optimizer = torch.optim.SGD(mlp_model.parameters(), lr=0.1)
all_epoch_losses = train(
    model=mlp_model,
    optimizer=optimizer,
    train_data_loaders=data.get_train_dataloader_epochs(256, 10),
    on_epoch_end=on_epoch_end,
)

fig, ax = plt.subplots()
plot_losses(ax, 
            [all_epoch_losses_scratch, all_epoch_losses], 
            ['From Scratch', 'PyTorch'])


# In[9]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(mlp_model, test_loader)
print(f'Test loss: {test_loss:.4f}')
accuracy = evaluate_accuracy(mlp_model, test_loader)
print(f'Test accuracy: {accuracy:.2%}')

test_loader = data.get_test_dataloader(data.test_size)
test_loss_scratch = evaluate_loss(mlp_model_scratch, test_loader)
print(f'Test loss (scratch): {test_loss_scratch:.4f}')
accuracy_scratch = evaluate_accuracy(mlp_model_scratch, test_loader)
print(f'Test accuracy (scratch): {accuracy_scratch:.2%}')


# In[10]:


(X, y) = next(iter(data.get_test_dataloader(batch_size=18)))


# In[11]:


y_hat = mlp_model.predict(X)
print('Predicted labels:', data.get_text_labels(y_hat.tolist()))
print('True labels:     ', data.get_text_labels(y.tolist()))
print(f'accuracy: {(y_hat == y).sum() / y.numel():.2%}')
y_hat_scratch = mlp_model_scratch.predict(X)
print('Predicted labels (scratch):', data.get_text_labels(y_hat_scratch.tolist()))
print(f'accuracy (scratch): {(y_hat_scratch == y).sum() / y.numel():.2%}')
