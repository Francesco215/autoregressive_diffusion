#%%
import io
import torch
from typing import Optional
from datasets import load_dataset
from edm2.cs_dataloading import OpenVidDataloader

batch_size = 16
dataloader = OpenVidDataloader(batch_size=batch_size, num_workers=32, device="cuda")
mean_sum = 0
std_sum = 0
num_batches = 0

for i, data in enumerate(dataloader):
    images = data["latents"].to("cuda")
    mean_sum += images.mean()
    std_sum += images.std()#dim=-3).mean()
    num_batches += i
    if i == 20:
        break

mean = mean_sum / num_batches
std = std_sum / num_batches

print(f"Mean: {mean}, Std: {std}")

# %%
