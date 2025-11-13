#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:


def cpu():  
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0): 
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def num_gpus():  
    """Return the number of available GPUs."""
    return torch.cuda.device_count()

def try_gpu(i=0):  
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

cpu(), gpu(), gpu(1), num_gpus(), try_gpu(), try_gpu(10)


# In[3]:


def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()


# In[4]:


x = torch.tensor([1, 2, 3])
x.device


# In[5]:


X = torch.ones(2, 3, device=try_gpu())
X


# In[6]:


Y = torch.rand(2, 3, device=try_gpu(1))
Y


# In[7]:


# Z = X.to(try_gpu(1))
# print(X)
# print(Z)


# In[8]:


net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())


# In[9]:


net[0].weight.data.device
