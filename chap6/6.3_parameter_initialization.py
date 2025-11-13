#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:


net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape


# In[3]:


def init_normal(module):
    if isinstance(module, nn.Linear):  
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]


# ## Pylance类型检查警告解释
# 
# 你看到的 `"__getitem__" method not defined on type "Module"` 警告是Pylance（Python类型检查器）的一个限制。
# 
# ### 问题原因：
# - `nn.Sequential` 继承自 `nn.Module`
# - Pylance只看到静态类型 `nn.Module`，不知道 `nn.Sequential` 实际上实现了 `__getitem__` 方法
# - 所以它认为 `net[0]` 是无效的索引操作
# 
# ### 实际情况：
# - 代码运行完全正常，因为 `nn.Sequential` 确实支持索引
# - 这只是一个静态类型检查的假阳性警告

# In[4]:


# 解决方案1: 使用类型注解明确指定类型
from typing import cast

net_typed = cast(nn.Sequential, net)
print("使用类型转换:")
print(f"第一层权重: {net_typed[0].weight.data[0]}")
print(f"第一层偏置: {net_typed[0].bias.data[0]}")

# 解决方案2: 使用 getattr 或 named_modules()
print("\n使用 named_modules():")
for name, module in net.named_modules():
    if isinstance(module, nn.Linear):
        print(f"模块 {name}: 权重形状 {module.weight.data.shape}")
        print(f"权重前几个值: {module.weight.data[0][:4]}")
        break

# 解决方案3: 使用列表访问（推荐用于教学）
print("\n使用 list() 转换:")
layers = list(net.children())
first_layer = layers[0]
print(f"第一层类型: {type(first_layer)}")
if isinstance(first_layer, nn.Linear):
    print(f"权重: {first_layer.weight.data[0][:4]}")
    print(f"偏置: {first_layer.bias.data[0]}")


# In[5]:


# 解决方案4: 最简单的方式 - 添加类型忽略注释
print("最简单的解决方案 - 使用类型忽略:")
print(f"权重: {net[0].weight.data[0][:4]}")  # type: ignore
print(f"偏置: {net[0].bias.data[0]}")        # type: ignore

# 解决方案5: 更好的类型安全写法（推荐用于生产代码）
def get_layer_params(sequential_net: nn.Sequential, layer_index: int):
    """安全地获取Sequential网络中指定层的参数"""
    layers = list(sequential_net.children())
    if layer_index >= len(layers):
        raise IndexError(f"Layer index {layer_index} out of range")
    
    layer = layers[layer_index]
    if isinstance(layer, nn.Linear):
        return layer.weight.data, layer.bias.data
    else:
        raise TypeError(f"Layer {layer_index} is not a Linear layer")

print("\n类型安全的访问:")
weight, bias = get_layer_params(net, 0)
print(f"权重前几个值: {weight[0][:4]}")
print(f"偏置: {bias[0]}")


# ## 总结：如何处理这个Pylance警告
# 
# ### 🎯 **推荐解决方案**（按使用场景）：
# 
# 1. **📚 学习/教学代码**：使用 `# type: ignore` 注释
#    ```python
#    net[0].weight.data[0]  # type: ignore
#    ```
# 
# 2. **🔧 生产代码**：使用类型安全的访问方法
#    ```python
#    layers = list(net.children())
#    first_layer = layers[0]
#    ```
# 
# 3. **📖 明确性**：使用 `cast()` 进行类型转换
#    ```python
#    net_typed = cast(nn.Sequential, net)
#    ```
# 
# ### ✅ **为什么原代码是正确的**：
# - `nn.Sequential` 确实实现了 `__getitem__` 方法
# - 代码在运行时完全正常
# - 这只是Pylance的静态分析限制
# 
# ### 🚫 **不需要修改原代码**：
# 你的原始代码 `net[0].weight.data[0]` 完全正确，只需要选择合适的方式消除警告即可。

# In[6]:


def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0]) # type: ignore
print(net[2].weight.data)


# In[7]:


def my_init(module):
    if isinstance(module, nn.Linear):
        print("Init", [(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2] # type: ignore


# In[8]:


net[0].weight.data[:] += 1 # type: ignore
net[0].weight.data[0, 0] = 42 # type: ignore
net[0].weight.data[0] # type: ignore

