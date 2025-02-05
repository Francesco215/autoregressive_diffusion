#%%
from edm2.dataloading import OpenVidDataloader, OpenVidDataset

# Create a dataloader
dataloader = OpenVidDataloader(batch_size=256, num_workers=8, device="cpu", dataset=OpenVidDataset())

#%%
# Get a batch from the dataloader
batch = next(iter(dataloader))

#%%
# Save batch to file 
import torch
torch.save(batch, "backup_batch.pt")
# %%
