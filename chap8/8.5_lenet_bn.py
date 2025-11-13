#!/usr/bin/env python
# coding: utf-8

# In[10]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[ ]:


import importlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

import importlib
from d2l.classification.model import LeNetBNClassifier, LeNetBNClassifierTorch, LeNetClassifierTorch
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


# In[12]:


rng = torch.Generator().manual_seed(42)


# In[13]:


data = FashionMNISTDataset()
train_data_loader = data.get_train_dataloader(batch_size=10)
train_iter = iter(train_data_loader)
X, y = next(train_iter)
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[14]:


from d2l import LeNetClassifierTorch

lenet = LeNetClassifierTorch(
    num_outputs=10,
    rng=rng
)

lenet.init((10, 1, 28, 28))
lenet.layer_summary((10, 1, 28, 28))


# 

# In[ ]:


lenet_bn = LeNetBNClassifierTorch(
    num_outputs=10,
    rng=rng
)
lenet_bn.init((10, 1, 28, 28))
lenet_bn.layer_summary((10, 1, 28, 28))


# In[ ]:


lenet_bn_s = LeNetBNClassifier(
    num_outputs=10,
    rng=rng
)
lenet_bn_s.init((10, 1, 28, 28))
lenet_bn_s.layer_summary((10, 1, 28, 28))


# In[17]:


def eval_on_epoch(model, epoch_id, batch_losses):
    test_loader = data.get_test_dataloader(batch_size=data.test_size)
    loss = evaluate_loss(model, test_loader)
    print(f'Epoch {epoch_id}, validation loss {loss:.4f}')
    accuracy = evaluate_accuracy(model, test_loader)
    print(f'Epoch {epoch_id}, validation accuracy {accuracy:.4f}')


# In[18]:


optimizer_lenet = torch.optim.Adam(lenet.parameters(), lr=0.005)
optimizer_lenet_bn = torch.optim.Adam(lenet_bn.parameters(), lr=0.005)
optimizer_lenet_bn_s = torch.optim.Adam(lenet_bn_s.parameters(), lr=0.005)


# In[19]:


epoch_losses = train(
    model=lenet,
    optimizer=optimizer_lenet,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)
fig, ax = plt.subplots()
plot_loss(ax, epoch_losses)


# In[20]:


epoch_losses = train(
    model=lenet_bn,
    optimizer=optimizer_lenet_bn,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)
fig, ax = plt.subplots()
plot_loss(ax, epoch_losses)


# In[21]:


epoch_losses = train(
    model=lenet_bn_s,
    optimizer=optimizer_lenet_bn_s,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)
fig, ax = plt.subplots()
plot_loss(ax, epoch_losses)


# In[22]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(lenet, test_loader)
print(f'Test loss: {test_loss:.4f}')
accuracy = evaluate_accuracy(lenet, test_loader)
print(f'Test accuracy: {accuracy:.2%}')


# In[23]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(lenet_bn, test_loader)
print(f'Test loss (bn) : {test_loss:.4f}')
accuracy = evaluate_accuracy(lenet_bn, test_loader)
print(f'Test accuracy (bn): {accuracy:.2%}')


# In[24]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(lenet_bn_s, test_loader)
print(f'Test loss (bn_s): {test_loss:.4f}')
accuracy = evaluate_accuracy(lenet_bn_s, test_loader)
print(f'Test accuracy (bn_s): {accuracy:.2%}')


# In[25]:


# save the models   
lenet_bn_s.save('lenet_bn_s.pth')
lenet_bn.save('lenet_bn.pth')
