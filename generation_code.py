#%%
from contextlib import nullcontext
from itertools import islice
import os
import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import matplotlib.pyplot as plt
from streaming.base.util import clean_stale_shared_memory


from edm2.sampler import edm_sampler_with_mse
from edm2.cs_dataloading import CsCollate, CsDataset, CsVaeCollate, CsVaeDataset
from edm2.networks_edm2 import UNet, Precond
import torch._dynamo.config

torch._dynamo.config.cache_size_limit = 100
torch.set_float32_matmul_precision('high')
clean_stale_shared_memory()

        
# vae = VAE.from_pretrained('s3://autoregressive-diffusion/saved_models/vae_cs_102354.pt').to(device)
# vae.mean=vae.mean.to(device)
# vae.std=vae.std.to(device)
vae = None
img_resolution = 64
device="cuda"

unet=UNet.from_pretrained(f'saved_models/unet_362.0M.pt')
unet_params = sum(p.numel() for p in unet.parameters())

unet=unet.to(device)

micro_batch_size = 4
batch_size = 4
accumulation_steps = batch_size//micro_batch_size
clip_length = 32
# training_steps = total_number_of_steps * batch_size
dataset = CsDataset(clip_size=clip_length, resolution=img_resolution, remote='s3://counter-strike-data/original/', local = f'../data/streaming_dataset/cs_diff_orig_a', batch_size=micro_batch_size, shuffle=False, cache_limit = '5000gb')
dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsCollate(), pin_memory=True, num_workers=1, shuffle=False, prefetch_factor=1)
steps_per_epoch = len(dataset)//micro_batch_size
n_epochs = 10
total_number_of_steps = n_epochs * steps_per_epoch
# total_number_of_steps = 40_000


sigma_data=1
precond = Precond(unet, use_fp16=False, sigma_data=sigma_data).to(device)
#%%

with torch.no_grad():
    latents, _ = next(islice(iter(dataloader),100))
    latents = latents.to(device)/.5

    #%%
    context = latents[:,:clip_length//2].to(device)
    precond.eval()
    sigma = torch.ones(context.shape[:2], device=device) * 0.05
    x, cache = precond(context, sigma, conditioning = None, update_cache=True)
    frames = context.clone()
    for _ in tqdm(range(clip_length//2)):
        x, _, _, cache= edm_sampler_with_mse(precond, cache=cache, conditioning = None, sigma_max = 80, sigma_min=0.01, num_steps=16, rho=7, guidance=1)
        frames = torch.cat((frames,x),dim=1)



    frames = ((frames*0.5).clip(-1,1)+1)*127.5
    frames = einops.rearrange(frames.long(), 'b t c h w -> b t h w c').cpu()


    #%%
    x = einops.rearrange(frames, 'b (t1 t2) h w c -> b (t1 h) (t2 w) c', t2=8)
    #set high resolution
    #%%
    # x = einops.rearrange(((latents*0.5).clip(-1,1)+1)*127.5, 'b (t1 t2) c h w -> b (t1 h) (t2 w) c', t2=8).long().cpu()
    plt.imshow(x[0])
    plt.axis('off')
    plt.savefig("cs.png",bbox_inches='tight',pad_inches=0, dpi=1000)

    # %%
