#!/usr/bin/env python
# coding: utf-8

# # 2.2. Data Preprocessing

# In[1]:


import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')


# In[2]:


import pandas as pd

data = pd.read_csv(data_file)
print(data)


# In[3]:


inputs = data.iloc[:, 0:2]
targets = data.iloc[:, 2]
print((inputs, targets))


# In[4]:


print(inputs)


# In[5]:


print(pd.get_dummies(inputs, dummy_na=True))
print(pd.get_dummies(inputs))
inputs = pd.get_dummies(inputs, dummy_na=True)


# In[6]:


inputs = inputs.fillna(inputs.mean())
print(inputs)


# In[7]:


import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print((X, y))


# # Exercises

# In[8]:


abalone = pd.read_csv(os.path.join('..', 'data', 'abalone.csv'), names=[ 'Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'])
print(abalone.head())


# In[9]:


abalone.isnull()


# In[10]:


abalone.isnull().sum(axis=0)


# In[11]:


missing_fraction = abalone.isnull().sum() / abalone.shape[0]
print(missing_fraction)


# In[12]:


num_cols = abalone.select_dtypes(include=['number']).columns
cat_cols = abalone.select_dtypes(include=['object', 'category']).columns

print(num_cols)
print(cat_cols)


# In[13]:


abalone.columns = ['Sex', 'Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']


# In[14]:


abalone


# In[ ]:




