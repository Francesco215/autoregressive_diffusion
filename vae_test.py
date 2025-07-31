#%%
import torch
from edm2.cs_dataloading import CsVaeCollate, CsVaeDataset
from edm2.vae import VAE
from torch.utils.data import DataLoader
import einops
device = "cuda"
# Hyperparameters
latent_channels = 8
n_res_blocks = 3
channels = [3, 16, 64, 256, latent_channels]

micro_batch_size = 2
batch_size = 8
accumulation_steps = batch_size//micro_batch_size
clip_length = 16

# Initialize models
vae = VAE.from_pretrained('s3://autoregressive-diffusion/saved_models/vae_cs_102354.pt').to(device)

clip_length = 16
micro_batch_size = 2
dataset = CsVaeDataset(clip_size=clip_length, remote='s3://counter-strike-data/vae_40M/', local = f'/data/streaming_dataset/cs_diff', batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsVaeCollate(), pin_memory=True, num_workers=4, shuffle=False)
means = next(iter(dataloader))[0]
for i, means in enumerate(dataloader):
    if i==10: break

# %%
with torch.no_grad():
    pixels = vae.decode(means[0][:,:8].to("cuda").to(torch.float).transpose(1,2), t=torch.tensor([0.1,0.1],device="cuda"))[0]
    pixels = einops.rearrange(pixels, 'b c t h w-> b t h w c')/2+.5
# %%
from matplotlib import pyplot as plt

plt.imshow(pixels.detach().cpu()[0,12])
# %%
