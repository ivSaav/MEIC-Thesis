from torch import nn
import torch

# def madgan_reconstruction(batch : torch.Tensor, netG : nn.Module, netD : nn.Module,
#                           device : str, max_iter : int = 100):
#     x = batch.to(device)
#     z = torch.zeros_like(x, requires_grad=True).to(device)
#     nn.init.normal_(z, std=0.05)
    
#     optm = torch.optim.RMSprop([z], lr=0.01)
#     loss_fn = nn.MSELoss(reduction='none')
    
#     previous_loss = float('inf')
#     for _ in range(max_iter):
#         optm.zero_grad()
#         G_z = netG(z)
#         print(G_z.shape)
#         loss = loss_fn(G_z, x).mean(dim=2)
#         print(loss.shape)
#         # if loss.mean() - previous_loss < 0.001: break # TODO
#         loss.backward()
#         optm.step()
#         # previous_loss = loss.mean()
        
#     return z, loss




def madgan_reconstruction(batch : torch.Tensor, netG : nn.Module, netD : nn.Module,
                          device : str, max_iter : int = 100):
    x = batch.to(device)
    z = torch.randn_like(x, requires_grad=True).to(device)
    # nn.init.normal_(z, std=0.05)
    
    optm = torch.optim.RMSprop([z], lr=0.01)
    loss_fn = nn.MSELoss(reduction='none')
    
    # norm_x = nn.functional.normalize(x, dim=1, p=2)
    for _ in range(max_iter):
        optm.zero_grad()
        G_z = netG(z)
        # norm_z = nn.functional.normalize(G_z, dim=1, p=2)
        # loss = loss_fn(G_z, x).sum(dim=(1,2))
        loss = loss_fn(G_z, x)
        
        # if loss.mean() - previous_loss < 0.001: break # TODO
        loss.mean().backward()
        optm.step()
        # previous_loss = loss.mean()
        
    return z, loss
 
        
def madgan_reconstruction_discr_loss(batch : torch.Tensor, netG : nn.Module,
                                     netD : nn.Module, device : str, max_iter : int = 100):
    x = batch.to(device)
    z = torch.zeros_like(x, requires_grad=True).to(device)
    nn.init.normal_(z, std=0.05)
    
    optm = torch.optim.RMSprop([z], lr=0.01)
    loss_fn = nn.MSELoss(reduction='none')
    
    for _ in range(max_iter):
        optm.zero_grad()
        G_z = netG(z)
        D_G_z = netD(G_z)
        loss = loss_fn(G_z, x)
        loss = (loss * D_G_z)
        loss.mean().backward()
        optm.step()
        
    return z, loss

def feature_anomaly_score(x, G_z, netD, device, lambda_thresh=0.1):
    # residual loss between x and G_z
    residual_loss = torch.abs(x - G_z)
    # loss calculation based on rich discrimination features from x and G_z
    x_features = netD(x.to(device))
    G_z_features = netD(G_z.to(device))
    
    # distance between x and G_z features
    discrimination_loss = torch.abs(x_features-G_z_features)
    total_loss =  (1-lambda_thresh)*residual_loss + lambda_thresh*discrimination_loss
    return total_loss

def mse_anomaly_score(x, G_z):
    return nn.functional.mse_loss(x, G_z, reduction='none')
    


def reconstruction_anomaly_scores(dataloader, netG : nn.Module, netD : nn.Module, device : str,
                                  max_iters=100, reconstr_fn=madgan_reconstruction):
    mse_scores = []
    for i, (x, filenames) in enumerate(dataloader, 0):
        _z, loss = reconstr_fn(x, netG, netD, device, max_iters)
        print(loss.sum(-1).shape)
        mse_scores.extend(zip(filenames, loss.detach().cpu().numpy()[0]))
        print(f"[{i}/{len(dataloader)}]", end='\r', flush=True)
        
    return mse_scores

