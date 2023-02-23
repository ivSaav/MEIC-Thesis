import torch 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, output_size : int, num_layers : int = 1, dropout : int  = 0, device : str = 'cpu'):
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
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size*2, num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size*4, num_layers, batch_first=True, dropout=dropout)
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*4, output_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        # create inputs
        batch_size, seq_len = x.size(0), x.size(1)
        h_0 = torch.zeros(1, batch_size, self.lstm0.hidden_size).to(self.device)
        c_0 = torch.zeros(1, batch_size, self.lstm0.hidden_size).to(self.device)
        
        recurr_features, (_h1, _c1) = self.lstm0(x, (h_0, c_0))
        recurr_features, _ = self.lstm1(recurr_features)
        recurr_features, _ = self.lstm2(recurr_features)
        
        outputs = self.linear(recurr_features.contiguous().view(batch_size*seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurr_features


    
class Discriminator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int = 1, dropout : int  = 0, device : str = 'cpu'):
        """LSTM Discriminator Model

        Args:
            input_size (int): size of input vector
            hidden_size (int): number of hidden units
            num_layers (int): number of stacked LSTM layers
            dropout (int, optional): dropout layer. Defaults to 0.
        """
        super().__init__()
        self.device = device
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # create inputs
        batch_size, seq_len = x.size(0), x.size(1)
        h_0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(self.device)
        
        recurr_features, (_h1, _c1) = self.lstm(x, (h_0, c_0))        
        outputs = self.linear(recurr_features.contiguous().view(batch_size*seq_len, self.lstm.hidden_size))
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