#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F


# In[2]:


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.mean()


# In[3]:


layer = CenteredLayer()
x = torch.tensor([1.0, 2, 3, 4, 5])
print(x.mean())
layer(x)


# In[4]:


net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())


# In[5]:


Y = net(torch.rand(4, 8))
Y.mean()


# In[6]:


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


# In[7]:


linear = MyLinear(5, 3)
print(linear.weight, linear.bias)
print(linear.weight.shape, linear.bias.shape)


# In[8]:


linear(torch.rand(2, 5))


# In[9]:


net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))


# In[ ]:




