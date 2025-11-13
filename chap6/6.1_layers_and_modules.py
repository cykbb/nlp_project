#!/usr/bin/env python
# coding: utf-8

# In[11]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[12]:


import torch
from torch import nn
from torch.nn import functional as F


# In[13]:


net = nn.Sequential(
    nn.Linear(20, 256), 
    nn.ReLU(), 
    nn.Linear(256, 10)
)

X = torch.rand(2, 20)
net(X).shape


# In[14]:


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.output(F.relu(self.hidden(X)))


# In[15]:


net = MLP()
net(X).shape


# In[16]:


class MySequential(nn.Module):
    def __init__(self, *args: nn.Module) -> None:
        super().__init__()
        self.modules_list = args
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for module in self.modules_list:
            X = module(X)
        return X


# In[17]:


net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape


# In[18]:


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# In[19]:


net = FixedHiddenMLP()
net(X)


# In[20]:


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)

