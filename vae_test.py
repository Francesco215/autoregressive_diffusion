#%%
import torch
from edm2.vae import VAE
#%%
vae = VAE(channels = [3,8,8,4], n_res_blocks=2, spatial_compressions=[1,2,4]).to("cpu")
#%%
x = torch.randn(2, 3, 12, 16, 16)
vae(x)[0].shape

# %%
