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

#%%

import torch
import einops
batch_size = 4
latents = torch.load("backup_batch.pt")['latents']
means, stds = latents.mean(dim=(0,1,-2,-1)), latents.std(dim=(-1,-2)).mean(dim=(0,1))
# %%

asd=((latents - means[None,None,:,None,None])/stds[None,None,:,None,None])
# %%
# plot the distrubuition of all of the possible values of asd
asd=asd.flatten()
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(asd.float().cpu().numpy(), bins=100, kde=True)
plt.show()

# %%
