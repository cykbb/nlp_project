#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[10]:


import torch
from d2l.regression.dataset import SyntheticRegressionDataset


# In[11]:


rng = torch.Generator().manual_seed(42)


# In[12]:


data = SyntheticRegressionDataset(w=torch.tensor([2, -3.4]), b=torch.tensor(4.2), num_test=100, num_train=1000, rng=rng)
((x_train, y_train), (x_test, y_test)) = (data.get_train_data(), data.get_test_data())
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[13]:


print('features:', data.X[0],'\nlabel:', data.y[0])


# In[14]:


for x_batch, y_batch in data.get_train_dataloader(batch_size=10):
    print(x_batch.shape, y_batch.shape)
    break


# In[15]:


for x_batch, y_batch in data.get_train_dataloader(batch_size=10):
    print(x_batch.shape, y_batch.shape)
    break

