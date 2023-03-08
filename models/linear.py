from torch import nn
from collections import OrderedDict

class Generator(nn.Module):
    def __init__(self, input_size : int, output_size : int, hidden_size : int, nlower : int = 4, neg_slope : float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.nlower = nlower
        
        def _block(prefix, idx, in_size, out_size):
            return [
                (f"{prefix}l{idx}", nn.Linear(in_size, out_size)),
                (f"{prefix}bn{idx}", nn.BatchNorm1d(out_size)),
                (f"{prefix}leaky{idx}", nn.LeakyReLU(neg_slope, inplace=True))
            ]
        
        layers = []
        if input_size != hidden_size:
            layers.extend(_block("in", "", input_size, hidden_size))
          
        # top inverted pyramid
        for i in range(1, nlower):
            layers.extend(_block("t", i, hidden_size//(2**(i-1)), hidden_size//(2**i)))
        
        # bottom pyramid
        top_out_size = layers[-2][1].num_features
        for i in range(1, nlower):
            layers.extend(_block("b", i, top_out_size*(2**(i-1)), top_out_size*(2**i)))
        
        # output layer
        if output_size != hidden_size:
            pyramid_out_size = layers[-2][1].num_features
            layers.append(("out", nn.Linear(pyramid_out_size, output_size)))
            layers.append(("out_bn", nn.BatchNorm1d(output_size)))
            
        self.main = nn.Sequential(OrderedDict(layers))
        # self.main.add_module('tanh', nn.C())
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.main(x)
        print(x)
        x = x.view(batch_size, self.output_size)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, nlayers : int = 4, neg_slope : float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.nlayers = nlayers
        
        self.main = nn.Sequential()
        # input layer
        if input_size != hidden_size:
            self.main.add_module('in', nn.Linear(input_size, hidden_size))
            self.main.add_module('in_bn', nn.BatchNorm1d(hidden_size))
            self.main.add_module('in_leaky', nn.LeakyReLU(neg_slope, inplace=True))
        
        # pyramid
        for i in range(1, nlayers):
            self.main.add_module(f'l{i}', nn.Linear(hidden_size//(2**(i-1)), hidden_size//(2**i)))
            self.main.add_module(f'bn{i}', nn.BatchNorm1d(hidden_size//(2**i)))
            self.main.add_module(f'leaky{i}', nn.LeakyReLU(neg_slope, inplace=True))
            
        pyramid_out_size = self.main.get_submodule(f'l{nlayers-1}').out_features
        self.main.add_module('out', nn.Linear(pyramid_out_size, 1))
        self.main.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.main(x)
        x = x.view(batch_size, 1)
        return x