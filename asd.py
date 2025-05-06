#%%
from diffusers import AutoencoderKL
import torch


vae2d = AutoencoderKL.from_pretrained("THUDM/CogView4-6B", subfolder="vae")
x = torch.randn(2,1024,16,16)                       # (B, C, T, H, W)
downsampler =vae2d.decoder.up_blocks[1].upsamplers[0]

# downsampler(x)
vae2d


# %%
