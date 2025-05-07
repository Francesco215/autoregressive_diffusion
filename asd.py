#%%
from diffusers import AutoencoderKL
import torch
from edm2.vae import VAE

vae2d = AutoencoderKL.from_pretrained("THUDM/CogView4-6B", subfolder="vae")
x = torch.randn(2,1024,16,16)                       # (B, C, T, H, W)
downsampler =vae2d.decoder.up_blocks[1].upsamplers[0]

# downsampler(x)
vae2d


# %%

vae = VAE(8)
vae._load_from_2d_model(vae2d)