#%%
import torch
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss
import logging
import os
os.environ['TORCH_LOGS']='recompiles'
os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

img_resolution = 64
img_channels = 16
unet = UNet(img_resolution=img_resolution, # Match your latent resolution
            img_channels=img_channels,
            label_dim = 0,
            model_channels=32,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=2,
            attn_resolutions=[16,8]
            )
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
precond = Precond(unet, use_fp16=False, sigma_data=1.).to("cuda")

# %%
torch.autograd.set_detect_anomaly(True, check_nan=True)

x = torch.randn(4, 10, img_channels, img_resolution, img_resolution, device="cuda")
# noise_level = torch.rand(2, 10, device="cuda")
# y = unet.forward(x, noise_level, None)
# y = precond.forward(x, noise_level, return_logvar=True)

loss=EDM2Loss()
y=loss(precond, x)

print(y,y.shape)

# %%
#this is to test if the unet is causal



# %%
