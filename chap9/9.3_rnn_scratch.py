#!/usr/bin/env python
# coding: utf-8

# In[11]:


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# In[12]:


from d2l.language_model.dataset import TimeMachineDataset


# In[13]:


timemachine = TimeMachineDataset(root="../data", num_steps=35)
timemachine.num_samples, timemachine.vocab.size()


# In[14]:


data = (iter(timemachine.get_train_dataloader(batch_size=32)))
for X, Y in data:
    print('X:', X.shape, '\n', X)
    print('Y:', Y.shape, '\n', Y)
    break


# In[15]:


import torch
import torch.nn as nn
rng = torch.Generator().manual_seed(0)


# In[16]:


from d2l.language_model.model import RNNLanguageModel


# In[17]:


from d2l.language_model.model import RNNLanguageModelTorch, LSTMLanguageModelTorch, LSTMLanguageModel

rnn_lm = RNNLanguageModel(
    input_size=timemachine.vocab.size(),
    hidden_size=32,
    num_layers=2
)
rnn_lm_torch = RNNLanguageModelTorch(
    input_size=timemachine.vocab.size(),
    hiddens_size=32,
    num_layers=2
)
lstm_lm = LSTMLanguageModel(
    input_size=timemachine.vocab.size(),
    hiddens_size=32,
    num_layers=2
)
lstm_lm_torch = LSTMLanguageModelTorch(
    input_size=timemachine.vocab.size(),
    hidden_size=32,
    num_layers=2
)


# In[18]:


from d2l.language_model.evaluator import evaluate_rnn_loss

def eval_on_epoch(model, epoch_id, batch_losses):
    loss = evaluate_rnn_loss(model, timemachine.get_test_dataloader(batch_size=timemachine.test_size))
    print(f'Epoch {epoch_id}, validation loss {loss:.4f}')


# In[19]:


from d2l.base.plot import plot_losses
from d2l.language_model.trainer import train_rnn

rnn_epoch_loss = train_rnn(
    model=rnn_lm,
    optimizer=torch.optim.Adam(rnn_lm.parameters(), lr=0.01),
    train_data_loaders=timemachine.get_train_dataloader_epochs(batch_size=4, num_epochs=10),
    on_epoch_end=eval_on_epoch,
    show_progress=False,
)

rnn_epoch_loss_torch = train_rnn(
    model=rnn_lm_torch,
    optimizer=torch.optim.Adam(rnn_lm_torch.parameters(), lr=0.01),
    train_data_loaders=timemachine.get_train_dataloader_epochs(batch_size=4, num_epochs=10),
    on_epoch_end=eval_on_epoch,
    show_progress=False,
)

lstm_epoch_loss = train_rnn(
    model=lstm_lm,
    optimizer=torch.optim.Adam(lstm_lm.parameters(), lr=0.01),
    train_data_loaders=timemachine.get_train_dataloader_epochs(batch_size=4, num_epochs=10),
    on_epoch_end=eval_on_epoch,
    show_progress=False,
)

lstm_epoch_loss_torch = train_rnn(
    model=lstm_lm_torch,
    optimizer=torch.optim.Adam(lstm_lm_torch.parameters(), lr=0.01),
    train_data_loaders=timemachine.get_train_dataloader_epochs(batch_size=4, num_epochs=10),
    on_epoch_end=eval_on_epoch,
    show_progress=False,
)


# In[23]:


from matplotlib import pyplot as plt

fig, ax = plt.subplots()
plot_losses(ax, [rnn_epoch_loss, rnn_epoch_loss_torch, lstm_epoch_loss, lstm_epoch_loss_torch], 
            labels=['RNN From Scratch', 'RNN Torch', 'LSTM From Scratch', 'LSTM Torch'])


# In[24]:


# save the models
torch.save(rnn_lm.state_dict(), 'rnn_lm_scratch.pth')
torch.save(rnn_lm_torch.state_dict(), 'rnn_lm_torch.pth')
torch.save(lstm_lm.state_dict(), 'lstm_lm_scratch.pth')
torch.save(lstm_lm_torch.state_dict(), 'lstm_lm_torch.pth')
