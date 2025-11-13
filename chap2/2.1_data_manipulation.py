#!/usr/bin/env python
# coding: utf-8

# # 2.1 Data Manipulation

# In[36]:


import torch


# In[37]:


x = torch.arange(12, dtype=torch.float32)
x


# In[38]:


x.numel()


# In[39]:


x.shape


# In[40]:


X = x.reshape(3, 4)
X


# In[41]:


torch.zeros((2, 3, 4))


# In[42]:


torch.ones((2, 3, 4))


# In[43]:


torch.randn(3, 4, dtype=torch.float32, generator=torch.Generator().manual_seed(2))


# In[44]:


torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


# In[45]:


X[-1], X[1:3]


# In[46]:


X[1, 2] = 17
X


# In[47]:


X[:2, :] = 12
X


# In[48]:


torch.exp(x)


# In[49]:


x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y


# In[50]:


X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([
    [2.0, 1, 4, 3], 
    [1, 2, 3, 4], 
    [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)


# In[51]:


X == Y


# In[52]:


X.sum()


# In[53]:


a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
a, b


# In[54]:


a + b


# In[55]:


before = id(Y)
Y = Y + X
id(Y) == before


# In[56]:


Z = torch.zeros_like(Y)
print(f'id(Z) = {id(Z)}')
Z[:] = X + Y
print(f'id(Z) = {id(Z)}')


# In[57]:


before = id(X)
X += Y
id(X) == before


# In[58]:


import numpy as np
A = np.array([1, 2, 3])
B = torch.tensor(A)
print((A, B))
B[1] = 4
print((A, B))
B.numpy()


# In[59]:


A = np.array([1, 2, 3])
B = torch.from_numpy(A)
print((A, B))
B[1] = 4
print((A, B))
B.numpy()


# In[60]:


a = torch.tensor([3.5])
a, a.item(), float(a), int(a)


# # Exercises

# In[61]:


X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
X == Y


# In[62]:


X < Y, X > Y


# In[63]:


X = torch.arange(12).reshape((3, 4))
Y = torch.arange(4).reshape((1, 4))
print((X, Y))
X + Y


# In[64]:


X = torch.arange(12).reshape((3, 4))
Y = torch.arange(4)
print((X, Y))
X + Y


# In[65]:


X = torch.arange(12).reshape((3, 4))
Y = torch.arange(3).reshape(3, 1)
print((X, Y))
X + Y


# In[66]:


X = torch.arange(12).reshape((3, 4))
Y = torch.arange(3).reshape((3, 1))
print((X, Y))
X + Y


# In[67]:


X = torch.arange(12).reshape((3, 4))
Y = torch.arange(4).reshape((4, 1))
print((X, Y))
X + Y

