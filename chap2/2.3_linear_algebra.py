#!/usr/bin/env python
# coding: utf-8

# In[662]:


import torch


# In[663]:


x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y


# In[664]:


x = torch.arange(3)
x


# In[665]:


x[2]


# In[666]:


len(x)


# In[667]:


x.shape


# In[668]:


A = torch.arange(6).reshape(3, 2)
A


# In[669]:


A.T


# In[670]:


A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T


# In[671]:


torch.arange(24).reshape(2, 3, 4)


# In[672]:


A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B


# In[673]:


A * B


# In[674]:


a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, a * X


# In[675]:


x = torch.arange(3, dtype=torch.float32)
x, x.sum()


# In[676]:


A


# In[677]:


print((A.shape, A.sum())) 
print((A.sum(dim=0), A.sum(dim=1)))


# In[678]:


A.shape, A.sum(dim=0).shape


# In[679]:


A.shape, A.sum(dim=0, keepdim=True).shape


# In[680]:


A.shape, A.sum(dim=1).shape


# In[681]:


A.shape, A.sum(dim=1, keepdim=True).shape


# In[682]:


A.sum(dim=[0, 1]) == A.sum()  # Same as A.sum()


# In[683]:


A.mean(), A.sum() / A.numel()


# In[684]:


A.mean(dim=0), A.sum(dim=0) / A.shape[0]


# In[685]:


sum_A = A.sum(dim=1, keepdim=True)
sum_A, sum_A.shape


# In[686]:


A


# In[687]:


A / sum_A


# In[688]:


A.cumsum(dim=0)


# In[689]:


y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)


# In[690]:


torch.sum(x * y)


# In[691]:


A, x


# In[692]:


A.shape, x.shape, torch.mv(A, x).shape, (A@x).shape


# In[693]:


B = torch.ones(3, 4)
torch.mm(A, B), A@B


# In[694]:


u = torch.tensor([3.0, -4.0])
torch.norm(u)


# In[695]:


torch.abs(u).sum()


# In[696]:


torch.norm(torch.ones((4, 9)))


# In[697]:


X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
len(X), X.numel()


# In[698]:


X, X.sum(dim=1)


# In[699]:


X.shape, X.sum(dim=1, keepdim=True).shape


# In[700]:


X / X.sum(dim=1, keepdim=True)


# In[701]:


A / A.sum(dim=1, keepdim=True)


# In[702]:


X


# In[703]:


X.sum(), X.sum(dim=0), X.sum(dim=1), X.sum(dim=2)


# In[704]:


X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
X


# In[705]:


torch.linalg.norm(X)  # Frobenius norm by default


# 先把尺寸写清楚：
# 
# * $A\in\mathbb R^{2^{10}\times 2^{16}}=1024\times 65536$
# * $B\in\mathbb R^{2^{16}\times 2^{5}}=65536\times 32$
# * $C\in\mathbb R^{2^{5}\times 2^{14}}=32\times 16384$
# * 结果 $ABC\in\mathbb R^{1024\times 16384}$
# 
# ## 计算量（以乘法次数计，FLOPs 约为它的 2 倍）
# 
# * $(AB)C$:
# 
#   * $AB:\ 1024\times 65536\times 32=2^{10+16+5}=2^{31}=2{,}147{,}483{,}648$
#   * $(AB)C:\ 1024\times 32\times 16384=2^{10+5+14}=2^{29}=536{,}870{,}912$
#   * 总计 $=2^{31}+2^{29}=5\times 2^{29}=2{,}684{,}354{,}560$ 乘法
#     （≈ **5.37×10⁹ FLOPs**）
# 
# * $A(BC)$:
# 
#   * $BC:\ 65536\times 32\times 16384=2^{16+5+14}=2^{35}=34{,}359{,}738{,}368$
#   * $A(BC):\ 1024\times 65536\times 16384=2^{10+16+14}=2^{40}=1{,}099{,}511{,}627{,}776$
#   * 总计 $=2^{35}+2^{40}=33\times 2^{35}=1{,}133{,}871{,}366{,}144$ 乘法
#     （≈ **2.27×10¹² FLOPs**）
# 
# **比值**：$\dfrac{1.1339\times 10^{12}}{2.684\times 10^{9}}\approx 422.4$。
# → $(AB)C$ 约 **快 400+ 倍**。
# 
# ## 中间结果的内存
# 
# * $(AB)C$ 的中间矩阵 $AB\in\mathbb R^{1024\times 32}$：元素数 $2^{10+5}=2^{15}=32{,}768$。
# 
#   * float32 ≈ **128 KB**（float64 ≈ 256 KB）。
# * $A(BC)$ 的中间矩阵 $BC\in\mathbb R^{65536\times 16384}$：元素数 $2^{16+14}=2^{30}=1{,}073{,}741{,}824$。
# 
#   * float32 ≈ **4 GB**（float64 ≈ 8 GB）。
# 
# **结论**：选择 $(AB)C$。它不仅计算量远小（\~422×），而且中间张量极小（\~128 KB vs 4–8 GB），更能命中缓存、也更可能在 GPU/内存中放得下。原因是矩阵连乘的代价取决于每一步的三维乘法尺寸 $a\times b\times c$；先把产生**较小中间维度**（这里是 $32$）的那两项相乘最划算。
# 

# In[706]:


A = torch.ones(100, 200)
B = torch.ones(100, 200)
C = torch.ones(100, 200)

(A.shape, B.shape, C.shape)


# In[707]:


torch.hstack((A, B, C)).shape


# In[708]:


torch.vstack((A, B, C)).shape


# In[714]:


torch.cat((A, B, C), dim=0).shape, torch.cat((A, B, C), dim=1).shape


# In[710]:


D = torch.cat((A, B, C))[100:200, :]
D.shape


# In[711]:


x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
x, y


# In[712]:


torch.cat((x, y)).shape

