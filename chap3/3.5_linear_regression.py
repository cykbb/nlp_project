#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[2]:


import torch


# In[3]:


import importlib
import d2l.regression.model as regression
from d2l.regression.dataset import SyntheticRegressionDataset
from d2l.regression.model import LinearRegression
from d2l.base.trainer import train
from d2l.base.evaluator import evaluate_loss
from d2l.base.optimizer import SGDOptimizer
from d2l.base.plot import plot_loss
from typing import List
from matplotlib import pyplot as plt

importlib.reload(regression)


# In[4]:


rng = torch.Generator().manual_seed(42)  


# In[5]:


regression_data = SyntheticRegressionDataset(w=torch.tensor([2, -3.4]), b=torch.tensor(4.2), rng=rng)


# In[6]:


regression = LinearRegression(num_features=2, rng=rng)
with torch.no_grad():
    print(f"Initial weights: {regression.w.reshape((-1,))}, bias: {regression.b}") 


# In[7]:


SGD_optimizer = SGDOptimizer(params=list(regression.parameters()), lr=0.03)
all_epoch_loss: List[List[float]] = train(
    model=regression,
    optimizer=SGD_optimizer,
    train_data_loaders=regression_data.get_train_dataloader_epochs(batch_size=32, epochs=7),
)
fig, ax = plt.subplots()
plot_loss(ax, all_epoch_loss)


# In[8]:


test_loss = evaluate_loss(regression, regression_data.get_test_dataloader(batch_size=regression_data.num_test))

print(f"Test loss: {test_loss:.6f}")
with torch.no_grad():
    print(f"Learned weights: {regression.w.reshape((-1,))}, expected: [2, -3.4]")
    print(f"Learned bias: {regression.b}, expected: 4.2")


# 
