#%%
from edm2.networks_edm2 import UNet, Precond

device = "cuda"
unet = UNet(img_resolution=64, # Match your latent resolution
            img_channels=4, # Match your latent channels
            label_dim = 4,
            model_channels=128,
            channel_mult=[1,2,3,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=3,
            video_attn_resolutions=[8],
            frame_attn_resolutions=[16],
            )
            
precond = Precond(unet, use_fp16=True, sigma_data=1.).to(device)

# %%
model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/'
model_name = 'edm2-img512-xs-2147483-0.135.pkl'

import dnnlib
import pickle
with dnnlib.util.open_url(f'{model_name}') as f:
    data = pickle.load(f)
net= data['ema'].to("cuda")

# %%

unet.load_from_2d(net.unet)
# %%
unet.save_to_state_dict("s3://autoregressive-diffusion/saved_models/edm2-img512-xs.pt")

# %%
