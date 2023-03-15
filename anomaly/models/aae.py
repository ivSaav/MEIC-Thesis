import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

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
            nn.BatchNorm1d(512),
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
            nn.BatchNorm1d(512),
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


# # Use binary cross-entropy loss
# adversarial_loss = torch.nn.BCELoss()
# pixelwise_loss = torch.nn.L1Loss()

# # Initialize generator and discriminator
# encoder = Encoder()
# decoder = Decoder()
# discriminator = Discriminator()

# if cuda:
#     encoder.cuda()
#     decoder.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#     pixelwise_loss.cuda()

# # Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# # Optimizers
# optimizer_G = torch.optim.Adam(
#     itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
# )
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits"""
#     # Sample noise
#     z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
#     gen_imgs = decoder(z)
#     save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# # ----------
# #  Training
# # ----------

# for epoch in range(opt.n_epochs):
#     for i, (imgs, _) in enumerate(dataloader):

#         # Adversarial ground truths
#         valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
#         fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

#         # Configure input
#         real_imgs = Variable(imgs.type(Tensor))

#         # -----------------
#         #  Train Generator
#         # -----------------

#         optimizer_G.zero_grad()

#         encoded_imgs = encoder(real_imgs)
#         decoded_imgs = decoder(encoded_imgs)

#         # Loss measures generator's ability to fool the discriminator
#         g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
#             decoded_imgs, real_imgs
#         )

#         g_loss.backward()
#         optimizer_G.step()

#         # ---------------------
#         #  Train Discriminator
#         # ---------------------

#         optimizer_D.zero_grad()

#         # Sample noise as discriminator ground truth
#         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = adversarial_loss(discriminator(z), valid)
#         fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
#         d_loss = 0.5 * (real_loss + fake_loss)

#         d_loss.backward()
#         optimizer_D.step()

#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#         )

#         batches_done = epoch * len(dataloader) + i
#         if batches_done % opt.sample_interval == 0:
#             sample_image(n_row=10, batches_done=batches_done)