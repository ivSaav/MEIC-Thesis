import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, output_size : int,
                 nlayers : int = 1, dropout : float = 0.0, bidirectional : bool = False, arch : str = 'LSTM'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nlayers = nlayers
        
        self.rnn = getattr(nn, arch)(input_size, hidden_size, nlayers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, nlayers : int = 1, dropout : float = 0.1, 
                 bidirectional : bool = False, arch : str = 'LSTM'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        
        self.rnn = getattr(nn, arch)(input_size, hidden_size, nlayers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return self.sigmoid(x)