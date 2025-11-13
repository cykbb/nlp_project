#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[2]:


import math
import torch
import numpy as np
import time

from d2l.base.plot import plot
import matplotlib.pyplot as plt


# In[3]:


n = 10000
a = torch.ones(n)
b = torch.ones(n)


# In[4]:


c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'{time.time() - t:.5f} sec'


# In[5]:


t = time.time()
d = a + b
f'{time.time() - t:.5f} sec'


# In[6]:


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)


# In[7]:


x = np.arange(-7, 7, 0.1)

parameters = [(0, 1), (0, 2), (3, 1)]
fig, ax = plt.subplots()
plot(ax, 
     (x, [normal(x, mu, sigma) for mu, sigma in parameters]),
     ('x', 'p(x)'),
     ((-7, 7), (0, 0.5)),
     ['normal(0, 1)', 'normal(0, 2)', 'normal(3, 1)']
)

