#%%
from diffusers import AutoencoderKL
import torch


vae2d = AutoencoderKL.from_pretrained("THUDM/CogView4-6B", subfolder="vae")
x = torch.randn(2,3,64,64)                       # (B, C, T, H, W)
downsampler =vae2d.decoder.up_blocks[1].upsamplers[0]

#%%

vae2d.encoder.conv_in.state_dict()

# %%
