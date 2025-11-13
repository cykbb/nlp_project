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
from d2l.classification.model import MLPClassifierDropout, MLPClassifierDropoutTorch
from d2l.classification.dataset import FashionMNISTDataset
from d2l.base.trainer import train
from d2l.base.evaluator import evaluate_accuracy, evaluate_loss
from d2l.base.optimizer import SGDOptimizer
from d2l.base.plot import plot_loss, show_images, plot_losses
from d2l.base.utils import mps
from typing import List
from matplotlib import pyplot as plt
import torch
import numpy as np


# In[3]:


rng = torch .random.manual_seed(42)


# In[4]:


data = FashionMNISTDataset()


# In[5]:


mlp_dropout = MLPClassifierDropout(
    num_features=784,
    num_outputs=10, 
    num_hiddens=[256, 256],
    dropouts=[0.2, 0.5],
    rng=rng
)
mlp_dropout_torch = MLPClassifierDropoutTorch(
    num_features=784,
    num_outputs=10, 
    num_hiddens=[256, 256],
    dropouts=[0.2, 0.5],
    rng=rng
)


# In[6]:


def eval_on_epoch(model, epoch_id, batch_losses):
    test_loader = data.get_test_dataloader(batch_size=data.test_size)
    loss = evaluate_loss(model, test_loader)
    print(f'Epoch {epoch_id}, validation loss {loss:.4f}')
    accuracy = evaluate_accuracy(model, test_loader)
    print(f'Epoch {epoch_id}, validation accuracy {accuracy:.4f}')


# In[ ]:


optimizer_scratch = SGDOptimizer(params=list(mlp_dropout.parameters()), lr=0.5)
epoch_losses = train(
    model=mlp_dropout,
    optimizer=optimizer_scratch,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)

optimizer_torch = torch.optim.SGD(params=mlp_dropout_torch.parameters(), lr=0.5)
epoch_losses_torch = train(
    model=mlp_dropout_torch,
    optimizer=optimizer_torch,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)

fig, ax = plt.subplots()
plot_losses(ax, [epoch_losses, epoch_losses_torch], labels=['From Scratch', 'PyTorch'])


# In[8]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(mlp_dropout, test_loader)
print(f'Test loss: {test_loss:.4f}')
accuracy = evaluate_accuracy(mlp_dropout, test_loader)
print(f'Test accuracy: {accuracy:.2%}')

test_loader = data.get_test_dataloader(data.test_size)
test_loss_scratch = evaluate_loss(mlp_dropout_torch, test_loader)
print(f'Test loss (torch): {test_loss_scratch:.4f}')
accuracy_scratch = evaluate_accuracy(mlp_dropout_torch, test_loader)
print(f'Test accuracy (torch): {accuracy_scratch:.2%}')
