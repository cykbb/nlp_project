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
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

import importlib
from d2l.classification.model import ResNetClassifier, ResNeXtClassifier
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

from d2l.base.function import corr2d, corr2d_multi_in, corr2d_multi_in_out, corr2d_multi_in_out_1x1, comp_conv2d, max_pool2d, avg_pool2d


# In[3]:


rng = torch.Generator().manual_seed(42)


# In[4]:


data = FashionMNISTDataset()
train_data_loader = data.get_train_dataloader(batch_size=10)
train_iter = iter(train_data_loader)
X, y = next(train_iter)
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[5]:


resnet = ResNetClassifier(
    num_outputs=10,
    rng=rng
)

resnet.init((10, 1, 28, 28))
resnet.layer_summary((10, 1, 28, 28))


# 

# In[6]:


resneXt = ResNeXtClassifier(
    num_outputs=10,
    rng=rng
)
resneXt.init((10, 1, 28, 28))
resneXt.layer_summary((10, 1, 28, 28))


# In[7]:


def eval_on_epoch(model, epoch_id, batch_losses):
    test_loader = data.get_test_dataloader(batch_size=data.test_size)
    loss = evaluate_loss(model, test_loader)
    print(f'Epoch {epoch_id}, validation loss {loss:.4f}')
    accuracy = evaluate_accuracy(model, test_loader)
    print(f'Epoch {epoch_id}, validation accuracy {accuracy:.4f}')


# In[8]:


optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=0.005)
optimizer_resnext = torch.optim.Adam(resneXt.parameters(), lr=0.005)


# In[9]:


epoch_losses = train(
    model=resnet,
    optimizer=optimizer_resnet,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)
fig, ax = plt.subplots()
plot_loss(ax, epoch_losses)


# In[10]:


epoch_losses = train(
    model=resneXt,
    optimizer=optimizer_resnext,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)
fig, ax = plt.subplots()
plot_loss(ax, epoch_losses)


# In[11]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(resnet, test_loader)
print(f'Test loss: {test_loss:.4f}')
accuracy = evaluate_accuracy(resnet, test_loader)
print(f'Test accuracy: {accuracy:.2%}')


# In[12]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(resneXt, test_loader)
print(f'Test loss (bn) : {test_loss:.4f}')
accuracy = evaluate_accuracy(resneXt, test_loader)
print(f'Test accuracy (bn): {accuracy:.2%}')


# In[13]:


# save the models
resnet.save('resnet.pth')
resneXt.save('resneXt.pth')
