#!/usr/bin/env python
# coding: utf-8

# In[21]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[22]:


import importlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

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

from d2l.base.function import corr2d, corr2d_multi_in, corr2d_multi_in_out, corr2d_multi_in_out_1x1, comp_conv2d, max_pool2d, avg_pool2d


# In[23]:


rng = torch.Generator().manual_seed(42)


# In[24]:


data = FashionMNISTDataset()
train_data_loader = data.get_train_dataloader(batch_size=10)
train_iter = iter(train_data_loader)
X, y = next(train_iter)
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[25]:


from d2l import LeNetClassifierTorch

lenet = LeNetClassifierTorch(
    num_outputs=10,
    rng=rng
)

lenet.init((10, 1, 28, 28))
lenet.layer_summary((10, 1, 28, 28))


# 

# In[26]:


def eval_on_epoch(model, epoch_id, batch_losses):
    test_loader = data.get_test_dataloader(batch_size=data.test_size)
    loss = evaluate_loss(model, test_loader)
    print(f'Epoch {epoch_id}, validation loss {loss:.4f}')
    accuracy = evaluate_accuracy(model, test_loader)
    print(f'Epoch {epoch_id}, validation accuracy {accuracy:.4f}')


# In[27]:


optimizer = torch.optim.Adam(lenet.parameters(), lr=0.005)


# In[28]:


epoch_losses = train(
    model=lenet,
    optimizer=optimizer,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, num_epochs=10),
    on_epoch_end=eval_on_epoch,
)
fig, ax = plt.subplots()
plot_loss(ax, epoch_losses)


# In[29]:


test_loader = data.get_test_dataloader(data.test_size)
test_loss = evaluate_loss(lenet, test_loader)
print(f'Test loss: {test_loss:.4f}')
accuracy = evaluate_accuracy(lenet, test_loader)
print(f'Test accuracy: {accuracy:.2%}')


# In[30]:


# save the model
lenet.save('lenet_fashion_mnist.pth')
