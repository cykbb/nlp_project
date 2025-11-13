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
from d2l.classification.model import SoftmaxClassifierTorch
from d2l.classification.dataset import FashionMNISTDataset
from d2l.base.trainer import train
from d2l.base.evaluator import evaluate_accuracy, evaluate_loss
from d2l.base.optimizer import SGDOptimizer
from d2l.base.plot import plot_loss, show_images
from d2l.base.utils import mps
from typing import List
from matplotlib import pyplot as plt


# In[3]:


import torch
device = mps()
print(f'Using device: {device}')


# In[4]:


X = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])
X.sum(dim=0, keepdim=True), X.sum(dim=1, keepdim=True)


# In[5]:


def softmax(X: torch.Tensor) -> torch.Tensor:
    return torch.exp(X) / torch.exp(X).sum(dim=1, keepdim=True)


# In[6]:


X = torch.rand((2, 5))
X


# In[7]:


X_prob = softmax(X)
X_prob, X_prob.sum(dim=1, keepdim=True)


# In[8]:


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]


# In[9]:


rng = torch.Generator().manual_seed(42)


# In[10]:


data = FashionMNISTDataset()
model = SoftmaxClassifierTorch(
    num_features=784, 
    num_outputs=10,
    rng=rng
)


# In[11]:


optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
all_epoch_losses = train(
    model=model,
    optimizer=optimizer,
    train_data_loaders=data.get_train_dataloader_epochs(batch_size=256, epochs=10),
)


# In[12]:


fig, ax = plt.subplots()
plot_loss(ax, all_epoch_losses)


# In[13]:


test_loader = data.get_test_dataloader(batch_size=256)
test_loss = evaluate_loss(model, test_loader)
print(f'Test loss: {test_loss:.4f}')
accuracy = evaluate_accuracy(model, test_loader)
print(f'Accuracy: {accuracy:.2%}')


# In[14]:


(X, y) = next(iter(data.get_test_dataloader(batch_size=18)))


# In[15]:


y_hat = model.predict(X)
print('Predicted labels:', data.get_text_labels(y_hat.tolist()))
print('True labels:     ', data.get_text_labels(y.tolist()))
print(f'accuracy: {(y_hat == y).sum() / y.numel():.2%}')
