from torch import nn
import torch



def covariance_reconstruction(batch : torch.Tensor, netG : nn.Module, netD : nn.Module,
                              device: str, max_iter : int = 100):
    x = batch.to(device)
    z = torch.rand_like(x, requires_grad=True).to(device)
    
    optm = torch.optim.RMSprop([z], lr=0.01)
    
    for _ in range(max_iter):
        optm.zero_grad()
        G_z = netG(z)
        cov_matrix = torch.cov(x, G_z)
        print(cov_matrix)

def madgan_reconstruction(batch : torch.Tensor, netG : nn.Module, netD : nn.Module,
                          device : str, max_iter : int = 100):
    x = batch.to(device)
    z = torch.rand_like(x, requires_grad=True).to(device)
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
    z = torch.rand_like(x, requires_grad=True).to(device)
    # nn.init.normal_(z, std=0.05)
    
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

def feature_anomaly_score(batch : torch.Tensor, netG : nn.Module, netD : nn.Module,
                          device, lambda_thresh=0.1, max_iter=100):
    x = batch.to(device)
    z = torch.rand_like(x, requiresgrad=True).to(device)
    # nn.init.normal(z, std=0.05)

    optm = torch.optim.RMSprop([z], lr=0.01)
    for _ in range(max_iter):
        optm.zero_grad()
        G_z = netG(z)
        # residual loss between x and G_z
        residual_loss = torch.abs(x - G_z)
        # loss calculation based on rich discrimination features from x and G_z
        x_features = netD(x.to(device))
        G_z_features = netD(G_z.to(device))

        # distance between x and G_z features
        discrimination_loss = torch.abs(x_features-G_z_features)
        total_loss =  (1-lambda_thresh)*residual_loss + lambda_thresh*discrimination_loss
        total_loss.mean().backward()
        optm.step()
    return z, total_loss


def mse_anomaly_score(x, G_z):
    return nn.functional.mse_loss(x, G_z, reduction='none')
    

