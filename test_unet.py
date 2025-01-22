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
            attn_resolutions=[64,32,16,8]
            ).to("cuda").to(torch.float16)
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
precond = Precond(unet, use_fp16=True, sigma_data=1.).to("cuda")

# %%
# x = torch.randn(4, 10, img_channels, img_resolution, img_resolution, device="cuda")
# # noise_level = torch.rand(2, 10, device="cuda")
# # y = unet.forward(x, noise_level, None)
# # y = precond.forward(x, noise_level, return_logvar=True)

# loss=EDM2Loss()
# y=loss(precond, x)

# # %%
# print(y.shape)

# %%

# x = torch.randn(2, 16, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
# x = torch.zeros(2, 8, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
# r = torch.randn(2, 8, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
# a = torch.cat((x,r),dim=1)
# x[0,4]=torch.randn(img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
# x = torch.cat((x,r),dim=1)
# noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=torch.float16)
# y = unet.forward(x, noise_level, None)-unet.forward(a, noise_level, None)



# %%
# x = torch.randn(2, 16, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
x = torch.zeros(2, 8, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
r = torch.randn(2, 8, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
a = torch.cat((x,r),dim=1)
x = torch.randn(2, 8, img_channels, img_resolution, img_resolution, device="cuda", dtype=torch.float16)
x = torch.cat((x,r),dim=1)
noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=torch.float16)
y = unet.forward(x, noise_level, None)-unet.forward(a, noise_level, None)


# %%
