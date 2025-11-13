#!/usr/bin/env python
# coding: utf-8

# In[49]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[50]:


import importlib
from d2l.classification.model import SoftmaxClassifier, SoftmaxClassifierLogSumExp
from d2l.classification.dataset import FashionMNISTDataset
from d2l.base.optimizer import SGDOptimizer
from d2l.base.plot import plot_loss, show_images
from typing import List
from matplotlib import pyplot as plt


# In[51]:


import time


# In[52]:


data = FashionMNISTDataset(resize=(32, 32))
len(data.train), len(data.test)


# In[53]:


print(data.train[0][0].shape, data.train[0][1])
(image, label) = data.train[0]
print(image.shape, label)


# In[54]:


X, y = next(iter(data.get_train_dataloader()))
print(X.shape, y.shape, X.dtype, y.dtype)


# In[55]:


tic = time.time()
for X, y in data.get_train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'


# In[56]:


X, y = next(iter(data.get_train_dataloader(batch_size=18)))
print(X.shape, y.shape)
show_images(X.reshape(18, 32, 32), titles=data.get_text_labels(y), layout=(3, 6))
