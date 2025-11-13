#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[2]:


import importlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from d2l.base.function import corr2d, corr2d_multi_in, corr2d_multi_in_out, corr2d_multi_in_out_1x1, comp_conv2d, max_pool2d, avg_pool2d


# In[3]:


X = torch.tensor([
    [0.0, 1.0, 2.0], 
    [3.0, 4.0, 5.0], 
    [6.0, 7.0, 8.0]
])

K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)


# In[4]:


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# In[5]:


X = torch.tensor([[[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])
K = torch.tensor([[[1.0, -1.0],
                  [1.0, -1.0]],
                  [[-1.0, 1.0],
                  [-1.0, 1.0]]])

Y = corr2d_multi_in(X, K)
Y


# In[6]:


X = torch.ones((6, 8))
X[:, 2:6] = 0
X


# In[7]:


K = torch.tensor([[1.0, -1.0]])


# In[8]:


Y = corr2d(X, K)
Y


# In[9]:


corr2d(X.t(), K)


# In[10]:


# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2).
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2 # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad # type: ignore
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')


# In[11]:


conv2d.weight.data.reshape((1, 2))


# In[12]:


conv2d = nn.LazyConv2d(1, kernel_size=3)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape


# In[13]:


# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape


# In[14]:


conv2d = nn.LazyConv2d(1, kernel_size=(5, 3))
comp_conv2d(conv2d, X).shape


# In[15]:


# We use a convolution kernel with height 5 and width 3. The padding on either
# side of the height and width are 2 and 1, respectively
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape


# In[16]:


conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
comp_conv2d(conv2d, X).shape


# In[17]:


conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape


# In[18]:


conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape


# In[19]:


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6


# In[20]:


X = torch.tensor([
    [0.0, 1.0, 2.0], 
    [3.0, 4.0, 5.0], 
    [6.0, 7.0, 8.0]
])
max_pool2d(X, (2, 2))


# ## 问题分析：MaxPool2d 的输入维度要求
# 
# PyTorch 的 `nn.MaxPool2d` 期望输入张量的形状为：
# - **3D**: (C, H, W) - 通道数、高度、宽度
# - **4D**: (N, C, H, W) - 批次大小、通道数、高度、宽度
# 
# 但当前 X 是 2D 张量 (3, 3)，需要添加维度。

# In[21]:


# 查看当前 X 的形状
print("X.shape:", X.shape)
print("X 的维度:", X.ndim)


# In[22]:


# 解决方案1：添加批次维度和通道维度 (N, C, H, W)
X_4d = X.reshape((1, 1, 3, 3))
print("添加维度后的形状:", X_4d.shape)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
result = pool2d(X_4d)
print("池化结果形状:", result.shape)
print("池化结果:\n", result)


# In[23]:


# 解决方案2：使用 unsqueeze 添加维度
X_4d_v2 = X.unsqueeze(0).unsqueeze(0)  # 在位置0添加两个维度
print("使用 unsqueeze 后的形状:", X_4d_v2.shape)

result_v2 = pool2d(X_4d_v2)
print("池化结果:\n", result_v2.squeeze())  # squeeze 去掉多余维度方便查看


# ### 总结
# 
# **错误原因**：
# - `nn.MaxPool2d` 需要 3D 或 4D 输入
# - 当前 X 是 2D 张量 (3, 3)
# 
# **解决方案**：
# 1. 使用 `reshape`: `X.reshape((1, 1, H, W))` - 显式指定新形状
# 2. 使用 `unsqueeze`: `X.unsqueeze(0).unsqueeze(0)` - 逐步添加维度
# 
# **池化过程**：
# - 输入: (1, 1, 3, 3)
# - 池化窗口: 3×3, padding=1, stride=2
# - 输出: (1, 1, 2, 2)
# - 结果选择了每个区域的最大值

# In[24]:


pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X.reshape(1, 1, 3, 3))


# In[25]:


X = torch.cat((X, X + 1), dim=0)
X.shape


# In[26]:


X


# In[27]:


pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X.reshape((1, 1, 6, 3)))

