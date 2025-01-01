#%%
import torch
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss
import logging

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
x = torch.randn(4, 43, 24, 16, 16, device="cuda")
# noise_level = torch.rand(2, 10, device="cuda")
# y = unet.forward(x, noise_level, None)
# y = precond.forward(x, noise_level, return_logvar=True)

loss=EDM2Loss()
y=loss(precond, x)

# %%
print(y.shape)

# %%

x = torch.randn(2, 16, 24, 64, 64, device="cuda", dtype=torch.float16)
noise_level = torch.randn(x.shape[:2], device="cuda", dtype=torch.float16)
y = unet.forward(x, noise_level, None)


# %%
