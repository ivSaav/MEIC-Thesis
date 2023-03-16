import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from typing import Tuple

from tools.viz import plot_data_values


def reparameterization(mu, logvar, l_dim, device):
    std = torch.exp(logvar / 2)
    sampled_z = torch.rand(mu.size(0), l_dim, device=device)
    # sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self, input_size, l_dim, device):
        super(Encoder, self).__init__()
        
        self.l_dim = l_dim
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, l_dim)
        self.logvar = nn.Linear(512, l_dim)

    def forward(self, x):
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.l_dim, self.device)
        return z


class Decoder(nn.Module):
    def __init__(self, output_size, l_dim):
        super(Decoder, self).__init__()
        
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(l_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_size),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, l_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(l_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


def train(netEnc : nn.Module, netDec : nn.Module, netD : nn.Module, 
          dataloader : torch.utils.data.DataLoader, opts, device, model_id : str = ""):
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    pixelwise_loss = nn.MSELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optmG = optim.Adam(
        itertools.chain(netEnc.parameters(), netDec.parameters()), 
        lr=opts.lr,
    )
    optmD = optim.Adam(netD.parameters(), lr=opts.lr)
    
    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    best_loss = np.inf

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(opts.epochs):
        D_epoch_loss = 0
        G_epoch_loss = 0
        
        # For each batch in the dataloader
        for _i, (x, _filename) in enumerate(dataloader):
            bsize = x.shape[0]
            real_labels = torch.full((bsize, 1), real_label, dtype=torch.float, device=device)
            fake_labels = torch.full((bsize, 1), fake_label, dtype=torch.float, device=device)
            
            real = x.to(device)
            # ===== Train Generator =====
            optmG.zero_grad()
            enc_x = netEnc(real)
            dec_x = netDec(enc_x)
            
            # Loss measures generator's ability to fool the discriminator
            errG = 0.001 * criterion(netD(enc_x), real_labels) + 0.999 * pixelwise_loss(
                dec_x, real
            )
            
            errG.backward()
            optmG.step()
            
            # ===== Train Discriminator =====
            optmD.zero_grad()
            # noise as discriminator ground truth
            z = torch.rand(bsize, opts.l_dim, device=device)
            
            # discriminator ability to classify real from generated samples
            outputs = netD(z)
            errD_real = criterion(netD(z), real_labels)
            errD_fake = criterion(netD(enc_x.detach()), fake_labels)    
            errD = 0.5 * (errD_real + errD_fake)
            D_x = outputs.mean().item()
            
            errD.backward()
            optmD.step()
            
            D_epoch_loss += errD.item() / bsize
            G_epoch_loss += errG.item() / bsize
        
        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                % (epoch, opts.epochs, errD.item(), errG.item(), D_x, errD_fake.mean().item()),
                end='\r', flush=True)
        D_losses.append(errD.item())
        G_losses.append(errG.item())
        
        if epoch % opts.sample_interval == 0:
            dec_x = dataloader.dataset.unscale(dec_x.detach().cpu().numpy())
            # fake = dataset.flatten(fake.detach().cpu().numpy())
            fig = plot_data_values(dec_x, 
                            title=f"Epoch {epoch} - G loss: {errG.item():.4f} - D loss: {errD.item():.4f}", 
                            labels=["B [G]", "alpha [deg]"],
                            scales={"alpha [deg]" : "linear"})
            preffix = f"{model_id}_" if model_id else ""
            plt.savefig(opts.model_out / "img" / f"{preffix}e{epoch}.png")
            plt.close(fig)
        if G_epoch_loss < best_loss and epoch > 0:
            best_loss = G_epoch_loss
            preffix = f"{model_id}_" if model_id else ""
            torch.save(netEnc.state_dict(), opts.model_out / f'{preffix}Enc.pth')
            torch.save(netDec.state_dict(), opts.model_out / f'{preffix}Dec.pth')
            torch.save(netD.state_dict(), opts.model_out / f'{preffix}D.pth')
    return G_losses, D_losses


def reconstruction_anomaly(netDec : nn.Module, netEnc : nn.Module, 
                           dataloader : torch.utils.data.DataLoader, device) -> Tuple[str, float]:
    scores = []
    loss_fn = nn.MSELoss(reduction="none")
    # calculate classification scores for each sample
    for i , (x, filenames) in enumerate(dataloader):
        x = x.to(device)
        enc_x = netEnc(x)
        dec_x = netDec(enc_x)
        
        errG = loss_fn(dec_x, x).mean(-1)
        errG = errG.detach().cpu().tolist()
        # calculate mean classification score for each sample
        for error, filename in zip(errG, filenames):
            scores.append((filename, error))
    return scores