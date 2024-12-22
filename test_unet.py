#%%
import torch
from edm2.networks_edm2 import UNet


unet = UNet(img_resolution=32,
            img_channels=12,
            label_dim = 0,
            model_channels=32,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=2,
            ).to("cuda").to(torch.float16)

# %%
# print number of unet parameters
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}")


# %%
x = torch.randn(2, 10, 12, 32, 32, device="cuda", dtype=torch.float16)
noise_level = torch.randn(1, device="cuda", dtype=torch.float16)
y = unet.forward(x, noise_level, None)

# %%
print(y.shape)

# %%
