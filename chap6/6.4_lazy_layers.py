#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:


net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))


# In[3]:


net[0].weight


# In[4]:


X = torch.rand(2, 20)
net(X)

net[0].weight.shape

