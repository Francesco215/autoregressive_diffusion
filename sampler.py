#%%
import torch
import numpy as np
from edm2.sampler import edm_sampler
#%%
from edm2.networks_edm2 import UNet, Precond
unet = UNet(img_resolution=16,
            img_channels=24,
            label_dim = 0,
            model_channels=32,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=2,
            ).to("cuda").to(torch.float16)
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
precond = Precond(unet, use_fp16=True, sigma_data=1., logvar_channels=128).to("cuda")


# %%
x = torch.randn(4, 17, 24, 16, 16, device="cuda")
# %%
y=edm_sampler(precond, x)
# %%
y.shape

# %%
