#!/usr/bin/env python
# coding: utf-8

# In[82]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[83]:


import importlib
from d2l.classification.model import MLPClassifierTorch
from d2l.classification.dataset import FashionMNISTDataset
from d2l.base.optimizer import SGDOptimizer
from d2l.base.plot import plot_loss, show_images
from d2l.base.utils import mps
from typing import List
from matplotlib import pyplot as plt
import torch


# In[84]:


from d2l import plot

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
fig, ax = plt.subplots(figsize=(5, 2.5))
plot(ax, (x.detach().numpy(), [y.detach().numpy()]), ('x', 'f(x)'), ((-8, 8), (0, 8)), ['relu(x)'])


# In[ ]:


y.backward(gradient=torch.ones_like(x))
fig, ax = plt.subplots(figsize=(5, 2.5))
plot(ax, (x.detach().numpy(), [x.grad.numpy()]), ('x', 'f\'(x)'), ((-8, 8), (0, 1)), ["relu'(x)"]) # type: ignore


# In[86]:


y = torch.sigmoid(x)
fig, ax = plt.subplots(figsize=(5, 2.5))
plot(ax, (x.detach().numpy(), [y.detach().numpy()]), ('x', 'f(x)'), ((-8, 8), (0, 1)), ['sigmoid(x)'])


# In[ ]:


x.grad.zero_() # type: ignore
y.backward(torch.ones_like(x))
fig, ax = plt.subplots(figsize=(5, 2.5))
plot(ax, (x.detach().numpy(), [x.grad.numpy()]), ('x', 'f\'(x)'), ((-8, 8), (0, 0.25)), ["sigmoid'(x)"]) # type: ignore


# In[88]:


y = torch.tanh(x)
fig, ax = plt.subplots(figsize=(5, 2.5))
plot(ax, (x.detach().numpy(), [y.detach().numpy()]), ('x', 'f(x)'), ((-8, 8), (-1, 1)), ['tanh(x)'])


# In[ ]:


x.grad.zero_() # type: ignore
y.backward(torch.ones_like(x))
fig, ax = plt.subplots(figsize=(5, 2.5))
plot(ax, (x.detach().numpy(), [x.grad.numpy()]), ('x', 'f\'(x)'), ((-8, 8), (0, 1)), ["tanh'(x)"]) # type: ignore
