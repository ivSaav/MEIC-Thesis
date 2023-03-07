from torch import nn

class Generator(nn.Module):
    def __init__(self, input_size : int, output_size : int, hidden_size : int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        self.l0 = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size*seq_len, self.input_size)
        x = self.l0(x)
        x = self.l1(x)
        x = x.view(batch_size, seq_len, self.output_size)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.l0 = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size*seq_len, self.input_size)
        x = self.l0(x)
        x = self.l1(x)
        x = x.view(batch_size, seq_len, 1)
        x = self.sigmoid(x)
        return x