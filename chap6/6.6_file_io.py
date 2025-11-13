#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F


# In[2]:


x = torch.arange(4)
torch.save(x, '../data/x-file')


# In[3]:


x2 = torch.load('../data/x-file')
x2


# In[4]:


y = torch.zeros(4)
torch.save([x, y],'../data/list-files')
x2, y2 = torch.load('../data/list-files')
(x2, y2)


# In[5]:


mydict = {'x': x, 'y': y}
torch.save(mydict, '../data/mydict')
mydict2 = torch.load('../data/mydict')
mydict2


# In[6]:


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)


# In[7]:


torch.save(net.state_dict(), '../data/mlp.params')


# In[8]:


clone = MLP()
clone.load_state_dict(torch.load('../data/mlp.params'))
clone.eval()


# In[9]:


Y_clone = clone(X)
Y_clone == Y


# In[ ]:




