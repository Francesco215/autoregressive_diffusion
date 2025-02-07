#%%
from tqdm import tqdm
from edm2.dataloading import OpenVidDataloader, OpenVidDataset
import einops
import torch

# Create a dataloader
dataloader = OpenVidDataloader(batch_size=128, num_workers=8, device="cpu", dataset=OpenVidDataset())
#%%
batch= next(iter(dataloader))

#%%
# Save batch to file 
import torch
torch.save(batch, "backup_batch.pt")

# %%
import torch
import einops
batch = torch.load("backup_batch.pt")
latents = batch["latents"]
latents = einops.rearrange(latents, "b t c h w -> (b t h w) c")
c_means = latents.mean(dim=1)
#%%
data = torch.load("cosmos_mean_var.pth")
c_means = data["means"]
cov = data["cov"]
latents = latents - c_means

# %%
c_cov =latents.t() @ latents / latents.size(0)
# %%
#plot covariance
import matplotlib.pyplot as plt
plt.imshow(c_cov.cpu())
#%%
plt.imshow(cov.cpu())
# %%

eigenvalues, eigenvectors = torch.linalg.eigh(cov.cpu())
# %%
plt.plot(eigenvalues, marker="o")
plt.yscale("log")
# %%
transformation_matrix = eigenvectors @ torch.diag(eigenvalues**-0.5)
# %%
transformed_latents = latents @ transformation_matrix
# %%
c_cov_transformed = transformed_latents.t() @ transformed_latents / transformed_latents.size(0)
# Plot covariance
import matplotlib.pyplot as plt
plt.imshow(c_cov_transformed.cpu())
# %%
