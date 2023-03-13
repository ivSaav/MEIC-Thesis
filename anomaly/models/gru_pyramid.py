import torch 
import torch.nn as nn

# Adapted from TAnoGAN
# https://github.com/mdabashar/TAnoGAN/blob/master/models/recurrent_models_pyramid.py

class Generator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, output_size : int, num_layers : int = 1, dropout : int  = 0, device : str = 'cpu',
                 activation : str = 'relu', bidirectional : bool = False):
        """LSTM Generator Model

        Args:
            input_size (int): size of input vector
            hidden_size (int): number of hidden units
            output_size (int): size of output vector
            num_layers (int): number of stacked LSTM layers
            dropout (int, optional): dropout layer. Defaults to 0.
        """
        super().__init__()
        self.device = device
        self.out_dim = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.l0 = nn.Linear(input_size, input_size//2)
        self.lstm0 = nn.GRU(input_size//2, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.lstm1 = nn.GRU(hidden_size, hidden_size*2, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.lstm2 = nn.GRU(hidden_size*2, hidden_size*4, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        self.actv = nn.Tanh()
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*4, output_size),
            self.actv
        )
        
    def forward(self, x):
        # create inputs
        batch_size, seq_len = x.size(0), x.size(1)
        
        outputs = self.l0(x)
        
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        recurr_features, hidden = self.lstm0(outputs, hidden)
        recurr_features, hidden = self.lstm1(recurr_features)
        recurr_features, _ = self.lstm2(recurr_features)
        
        outputs = self.linear(recurr_features.contiguous().view(batch_size*seq_len, self.hidden_size*4))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurr_features


    
class Discriminator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int = 1, dropout : int  = 0, device : str = 'cpu',
                 bidirectional : bool = False):
        """LSTM Discriminator Model

        Args:
            input_size (int): size of input vector
            hidden_size (int): number of hidden units
            num_layers (int): number of stacked LSTM layers
            dropout (int, optional): dropout layer. Defaults to 0.
        """
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.l0 = nn.Linear(input_size, input_size//2)
        self.lstm = nn.GRU(input_size//2, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.lstm1 = nn.GRU(hidden_size, hidden_size*2, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # create inputs
        batch_size, seq_len = x.size(0), x.size(1)
        outputs = self.l0(x)
        
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        recurr_features, hidden = self.lstm(outputs, hidden)
        recurr_features, _ = self.lstm1(recurr_features)
        outputs = self.linear(recurr_features.contiguous().view(batch_size*seq_len, self.hidden_size*2))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs, recurr_features
    
if __name__ == '__main__':
    batch_size = 1
    noise_dim = 100
    seq_dim = 4
    
    device = 'cpu'

    G = Generator(
        input_size=noise_dim,
        output_size=noise_dim,
        hidden_size=32,
        device=device
    )
    
    D = Discriminator(
        input_size=noise_dim,
        hidden_size=100,
        device=device
    )
    
    noise = torch.randn(1, 1, noise_dim)
    G_out, _ = G(noise)
    D_out, _ = D(G_out)
    
    print(G_out)
    print(D_out)

    print("Noise: ", noise.size())
    print("Generator output: ", G_out.size())
    print("Discriminator output: ", D_out.size())