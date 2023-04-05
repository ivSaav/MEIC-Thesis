import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from torch import optim

from typing import Tuple
import random
import gc as gc

from tools.viz import plot_data_values

def init_weights(model : nn.Sequential, slope=0.2):
    # Init weights with xavier uniform distribution to reduce vanishing gradients
    # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    for idx, layer in enumerate(model):
        if layer._get_name() == "Linear":
            if idx+1 >= len(model): continue # last layer
            
            actv = model[idx+1]._get_name()
            if "BatchNorm" in actv: actv = model[idx+2]._get_name()
            
            if actv == "LeakyReLU":
                nn.init.kaiming_normal_(layer.weight, a=slope)
            elif actv == "Sigmoid":
                nn.init.xavier_normal_(layer.weight, 1)
            elif actv == "Tanh":
                nn.init.xavier_normal_(layer.weight, 5/3)


class Encoder(nn.Module):
    def __init__(self, input_size, l_dim, hidden_size, device, slope=0.2):
        super(Encoder, self).__init__()
        
        self.l_dim = l_dim
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//16, 2),
        )

        init_weights(self.model, slope)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, output_size, l_dim, hidden_size, slope=0.2):
        super(Decoder, self).__init__()
        
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(l_dim, hidden_size),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size*4, hidden_size*8),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size*8, hidden_size*16),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size*16, output_size),
            nn.Sigmoid(),
        )
        
        init_weights(self.model, slope)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, l_dim, hidden_size, slope=0.2):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(l_dim, hidden_size),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size, hidden_size//4),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.LeakyReLU(slope, inplace=True),
            
            nn.Linear(hidden_size//16, 1),
            nn.Sigmoid(),
        )
        init_weights(self.model, slope)

    def forward(self, z):
        return self.model(z)
    

def gaussian_sample(nsamples, gopts):
    # determine x and y coordinates of the target mixture of gaussians
    x_centroid = (gopts.radius * np.sin(np.linspace(0, 2 * np.pi, gopts.nclusters, endpoint=False)) + 1) / 2
    y_centroid = (gopts.radius * np.cos(np.linspace(0, 2 * np.pi, gopts.nclusters, endpoint=False)) + 1) / 2
    # determine each gaussians mean (centroid) and standard deviation
    centroids = np.vstack([x_centroid, y_centroid]).T
    # determine the number of samples to be created per gaussian
    samples_per_gaussian = nsamples//gopts.nclusters

    target_z = np.empty((0, gopts.dim))
    # iterate over the number of distinct gaussians
    
    point_distr_indexes = set(random.sample(range(gopts.nclusters), nsamples % gopts.nclusters))
    for i, mu in enumerate(centroids):
        # place an additional point in current cluster in case of odd nsamples
        sp = samples_per_gaussian if i not in point_distr_indexes else samples_per_gaussian + 1
        # create n gaussian groups
        z_samples = np.random.normal(mu, gopts.sigma, size=(sp, gopts.dim))
        target_z = np.vstack([target_z, z_samples])  
    return target_z, centroids



def train(netEnc : nn.Module, netDec : nn.Module, netD : nn.Module, 
          dataloader : torch.utils.data.DataLoader, opts, gopts, device):
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    reconstr_loss = nn.MSELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    optmEnc = optim.Adam(netEnc.parameters(), lr=opts.G_lr)
    optmDec = optim.Adam(netDec.parameters(), lr=opts.G_lr)
    optmD = optim.Adam(netD.parameters(), lr=opts.D_lr)
    
    # Training Loop
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    best_loss = np.inf

    # For each epoch
    print("Starting Training Loop...")
    for epoch in range(opts.epochs):
        D_epoch_loss = 0
        G_epoch_loss = 0
        epoch_reconstr_loss = 0
        D_class = 0
        epoch_enc_x = np.empty((0, opts.l_dim))
        epoch_filenames = []
        
        # For each batch in the dataloader
        for i, (x, filenames) in enumerate(dataloader):
            bsize = x.shape[0]
            real_labels = torch.full((bsize, 1), real_label, dtype=torch.float, device=device)
            fake_labels = torch.full((bsize, 1), fake_label, dtype=torch.float, device=device)
            
            real = x.to(device)
            # ===== Train Generator =====
            optmEnc.zero_grad(), optmDec.zero_grad()
            enc_x = netEnc(real)
            dec_x = netDec(enc_x)
            
            errG = (reconstr_loss(dec_x, real) + criterion(netD(enc_x), real_labels)) * 0.5
            errG.backward()
            optmEnc.step(), optmDec.step()
            epoch_reconstr_loss += errG.item()
            
            # ===== Train Discriminator =====
            optmD.zero_grad()
            # noise as discriminator ground truth
            z_sample, _ = gaussian_sample(bsize, gopts)
            z = torch.from_numpy(z_sample).float().to(device)
            # discriminator ability to classify real from generated samples
            outputs = netD(z)
            errD_real = criterion(outputs, real_labels)
            errD_fake = criterion(netD(enc_x.detach()), fake_labels)    
            errD = (errD_real  + errD_fake) * 0.5
            D_class += outputs.mean().item()
            epoch_enc_x = np.vstack((epoch_enc_x, enc_x.detach().cpu().numpy()))
            epoch_filenames.extend(filenames)
            
            errD.backward()
            optmD.step()
            D_epoch_loss += errD.item()
            G_epoch_loss += errG.item()
        
        D_epoch_loss /= i
        G_epoch_loss /= i
        epoch_reconstr_loss /= i
        D_class /= i
            
        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(z): %.4f\tDec_Err: %.4f'
                % (epoch, opts.epochs, D_epoch_loss, G_epoch_loss, D_class, epoch_reconstr_loss),
                end='\r', flush=True)
        D_losses.append(errD.item())
        G_losses.append(errG.item())
        
        if epoch % opts.sample_interval == 0 or epoch == opts.epochs-1:
            dec_x = dataloader.dataset.unscale(dec_x.detach().cpu().numpy())
            # fake = dataset.flatten(fake.detach().cpu().numpy())
            fig = plot_data_values(dec_x, 
                            title=f"Epoch {epoch} - G loss: {errG.item():.4f} - D loss: {errD.item():.4f}", 
                            labels=["B [G]", "alpha [deg]"],
                            scales={"alpha [deg]" : "symlog"})
            plt.savefig(opts.model_out / "img" / f"e{epoch}.png")
            plt.close(fig)
            
            # save scatter of latent representation by the encoder
            fig, ax = plt.subplots(figsize=(10,10))
            plt.scatter(epoch_enc_x[:, 0], epoch_enc_x[:, 1], s=1)
            plt.savefig(opts.model_out / "img" / f"scatter{epoch}.png")
            plt.close(fig)
            
            gc.collect()
            
        if G_epoch_loss < best_loss and epoch > 0:
            best_loss = G_epoch_loss
            torch.save(netEnc.state_dict(), opts.model_out / 'Enc.pth')
            torch.save(netDec.state_dict(), opts.model_out / 'Dec.pth')
            torch.save(netD.state_dict(), opts.model_out / 'D.pth')



