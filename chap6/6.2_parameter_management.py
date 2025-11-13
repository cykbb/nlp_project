#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:


net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape


# In[3]:


net[2].state_dict()


# In[4]:


type(net[2].bias), net[2].bias.data


# In[5]:


net[2].weight.grad is None


# In[6]:


[(name, param.shape) for name, param in net.named_parameters()]


# In[7]:


# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0]) # type: ignore
net[2].weight.data[0, 0] = 100 # type: ignore
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0]) # type: ignore

