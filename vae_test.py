#%%
import torch
from edm2.vae import VAE
#%%
device = "cuda"
# Hyperparameters
latent_channels = 8
n_res_blocks = 3
channels = [3, 16, 64, 256, latent_channels]

# Initialize models
vae = VAE(channels = channels, n_res_blocks=n_res_blocks, spatial_compressions=[1,2,2,2], time_compressions=[1,2,2,1], logvar_mode=0.1).to(device)
#%%
x = torch.randn(2, 3, 32, 16, 16, device= "cuda")
vae(x)[0].shape

# %%
