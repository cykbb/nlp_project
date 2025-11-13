from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import d2l.base.function as d2l_F
from abc import abstractmethod

class RNNBlock(nn.Module):
    def __init__(self,
                 num_input: int,
                 num_hiddens: int) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_hiddens = num_hiddens
        self.W_xh = nn.Parameter(
            torch.randn(num_input, num_hiddens) * 0.01
        )
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * 0.01
        )
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))
        
    # assuming input shape is (time_steps, batch_size, num_input)
    # output is (time_steps, batch_size, num_hiddens) and the final state is (batch_size, num_hiddens)
    def forward(self, 
                input: torch.Tensor, 
                state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, batch_size, _ = input.shape
        if state is None:
            state = torch.zeros((batch_size, self.num_hiddens))
        output = []
        for X in input:
            H = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h
            )
            output.append(H)
            state = H
        return torch.stack(output, dim=0), state
    
class RNNsBlock(nn.Module):
    def __init__(self, 
                 embedding_dim: int,
                 num_hiddens: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_input = embedding_dim
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.rnn_blocks = nn.ModuleList()
        for i in range(num_layers):
            input_size = embedding_dim if i == 0 else num_hiddens
            self.rnn_blocks.append(RNNBlock(input_size, num_hiddens))
            
    # assuming input shape is (time_steps, batch_size, num_input)
    # states is a tensor of shape (num_layers, batch_size, num_hiddens), state in each layer is a tensor of shape (batch_size, num_hiddens)
    def forward(self, 
                input: torch.Tensor, 
                states: Optional[torch.Tensor] = None):
        _, batch_size, _ = input.shape
        if states is None:
            states = torch.zeros((self.num_layers, batch_size, self.num_hiddens))
        layer_outputs = []
        last_states = []
        current_layer_input = input
        for i in range(self.num_layers):
            state, rnn = states[i], self.rnn_blocks[i]
            output, last_state = rnn(current_layer_input, state)
            layer_outputs.append(output)
            last_states.append(last_state)
            current_layer_input = output
        return torch.stack(layer_outputs, dim=0), torch.stack(last_states, dim=0) 
        # output shape: (num_layers, time_steps, batch_size, num_hiddens), states shape: (num_layers, batch_size, num_hiddens)
        
class LSTMBlock(nn.Module):
    def __init__(self,
                 num_input: int,
                 num_hiddens: int) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_hiddens = num_hiddens
        self.W_xi = nn.Parameter(
            torch.randn(num_input, num_hiddens) * 0.01
        )
        self.W_hi = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * 0.01
        )
        self.b_i = nn.Parameter(torch.zeros(num_hiddens))
        
        self.W_xf = nn.Parameter(
            torch.randn(num_input, num_hiddens) * 0.01
        )
        self.W_hf = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * 0.01
        )
        self.b_f = nn.Parameter(torch.zeros(num_hiddens))
        
        self.W_xo = nn.Parameter(
            torch.randn(num_input, num_hiddens) * 0.01
        )
        self.W_ho = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * 0.01
        )
        self.b_o = nn.Parameter(torch.zeros(num_hiddens))
        
        self.W_xc = nn.Parameter(
            torch.randn(num_input, num_hiddens) * 0.01
        )
        self.W_hc = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * 0.01
        )
        self.b_c = nn.Parameter(torch.zeros(num_hiddens))
        
    # assuming input shape is (time_steps, batch_size, num_input)
    # output is (time_steps, batch_size, num_hiddens) and the final state is (batch_size, num_hiddens)
    def forward(self, 
                input: torch.Tensor, 
                states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _, batch_size, _ = input.shape
        if states is None:
            H = torch.zeros((batch_size, self.num_hiddens))
            C = torch.zeros((batch_size, self.num_hiddens))
        else:
            H, C = states
        output = []
        for X in input:
            I = torch.sigmoid(
                torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i
            )
            F = torch.sigmoid(
                torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f
            )
            O = torch.sigmoid(
                torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o
            )
            C_tilde = torch.tanh(
                torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c
            )
            C = F * C + I * C_tilde
            H = O * torch.tanh(C)
            output.append(H)
        return torch.stack(output, dim=0), (H, C)
    
class LSTMsBlock(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_input = input_size
        self.num_layers = num_layers
        self.num_hiddens = hidden_size
        self.lstm_blocks = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.lstm_blocks.append(LSTMBlock(input_size, hidden_size))
            
    # assuming input shape is (time_steps, batch_size, num_input)
    # states is a tuple of two tensors each of shape (num_layers, batch_size, num_hiddens), state in each layer is a tuple of two tensors each of shape (batch_size, num_hiddens)
    def forward(self, 
                input: torch.Tensor, 
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        _, batch_size, _ = input.shape
        if states is None:
            H = torch.zeros((self.num_layers, batch_size, self.num_hiddens))
            C = torch.zeros((self.num_layers, batch_size, self.num_hiddens))
        else:
            H, C = states
        layer_outputs = []
        last_states_H = []
        last_states_C = []
        current_layer_input = input
        for i in range(self.num_layers):
            state = (H[i], C[i])
            lstm = self.lstm_blocks[i]
            output, (last_state_H, last_state_C) = lstm(current_layer_input, state)
            layer_outputs.append(output)
            last_states_H.append(last_state_H)
            last_states_C.append(last_state_C)
            current_layer_input = output
        return torch.stack(layer_outputs, dim=0), (torch.stack(last_states_H, dim=0), torch.stack(last_states_C, dim=0))
        # output shape: (num_layers, time_steps, batch_size, num_hiddens), states shape: tuple of two tensors each of shape (num_layers, batch_size, num_hiddens)
        
