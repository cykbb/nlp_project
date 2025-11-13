#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[2]:


from d2l.language_model.dataset import TimeMachineDataset


# In[3]:


timemachine = TimeMachineDataset(root="../data", num_steps=35)
print(timemachine.num_samples, timemachine.vocab.size())
print(timemachine.train_size, timemachine.test_size)
print(timemachine.X_train.shape, timemachine.Y_train.shape)
print(timemachine.X_test.shape, timemachine.Y_test.shape)


# In[4]:


data = (iter(timemachine.get_train_dataloader(batch_size=32)))
for X, Y in data:
    print('X:', X.shape, '\n', X)
    print('Y:', Y.shape, '\n', Y)
    break

