#%%
import numpy as np
import torch
from tqdm import tqdm
from edm2.loss_weight import MultiNoiseLoss
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss
import logging
import os
from edm2.dataloading import OpenVidDataloader, RandomDataset#, OpenVidDataset
os.environ['TORCH_LOGS']='recompiles'
os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

img_resolution = 64
img_channels = 16
micro_batch_size = 2
num_workers = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = OpenVidDataloader(micro_batch_size, num_workers, device, dataset = RandomDataset())
#%%
unet = UNet(img_resolution=64, # Match your latent resolution
            img_channels=16, # Match your latent channels
            label_dim = dataloader.text_embedding_dim,
            model_channels=64,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=2,
            attn_resolutions=[16,8]
            )
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
precond = Precond(unet, use_fp16=True, sigma_data=1.).to("cuda")
precond.load_state_dict(torch.load("model_batch_2500.pt")["model_state_dict"])

# %%
# torch.autograd.set_detect_anomaly(True, check_nan=True)
batch = torch.load("backup_batch.pt")
s=58
micro_batch_size = 4
latents = batch['latents'][s:s+micro_batch_size].to(device)
text_embeddings = batch['text_embeddings'][s:s+micro_batch_size].to(device)
# x = torch.randn(4, 10, img_channels, img_resolution, img_resolution, device="cuda")
max_noise = 1e2
min_noise = 1e2
noise_level = torch.rand(latents.shape[:2], device=device)*(max_noise-min_noise)+min_noise

latents, noise_level, text_embeddings = latents.to(torch.float16), noise_level.to(torch.float16), text_embeddings.to(torch.float16)

loss=EDM2Loss(noise_weight=MultiNoiseLoss())
with torch.no_grad():
    for s in tqdm(np.logspace(-2,2,20)):
        sigma_shape = latents.shape[:2]
        sigma = torch.cat((torch.ones(sigma_shape)*0.05,torch.ones(sigma_shape)*s), dim=1).to(device)
        y=loss(precond, latents, sigma=sigma)

print(y)

# %%
loss.noise_weight.fit_loss_curve()
loss.noise_weight.plot('asd.png')
#this is to test if the unet is causal



# %%
unet.out_gain


# %%
