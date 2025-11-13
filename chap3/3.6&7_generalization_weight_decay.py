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
from d2l.regression.model import LinearRegression, LinearRegressionL2
from d2l.regression.dataset import SyntheticRegressionDataset
from d2l.base.trainer import train
from d2l.base.evaluator import evaluate_loss
from matplotlib import pyplot as plt
from d2l.base.plot import plot_losses
from d2l.base.optimizer import SGDOptimizer

importlib.reload(regression)


# In[4]:


rng = torch.Generator().manual_seed(42)


# In[5]:


w = torch.ones(200) * 0.01
b = torch.ones(1) * 0.05


# In[6]:


reg_data = SyntheticRegressionDataset(w=w, b=b, rng=rng, num_train=20, num_test=200)


# In[7]:


model = LinearRegression(num_features=200, rng=rng)
model_l2 = LinearRegressionL2(num_features=200, weight_decay=3, rng=rng)


# In[8]:


optimizer = SGDOptimizer(list(model.parameters()), lr=0.003)
optimizer_l2 = SGDOptimizer(list(model_l2.parameters()), lr=0.003)
all_epochs_losses = train(
    model=model,
    optimizer=optimizer,
    train_data_loaders=reg_data.get_train_dataloader_epochs(32, 100),
)
all_epochs_losses_l2 = train(
    model=model_l2,
    optimizer=optimizer_l2,
    train_data_loaders=reg_data.get_train_dataloader_epochs(32, 100),
)
fig, ax = plt.subplots()
plot_losses(ax, [all_epochs_losses, all_epochs_losses_l2], labels=['Without weight decay', 'With weight decay'])


# In[9]:


mse = evaluate_loss(model, reg_data.get_test_dataloader(batch_size=reg_data.num_test))
mse_l2 = evaluate_loss(model_l2, reg_data.get_test_dataloader(batch_size=reg_data.num_test))

print(f"Test MSE without L2 regularization: {mse:.6f}")
print(f"Test MSE with L2 regularization: {mse_l2:.6f}")
