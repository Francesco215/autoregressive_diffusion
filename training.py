#%%
from abc import abstractmethod
import os
import einops
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss
from edm2.loss_weight import MultiNoiseLoss

os.environ['DISABLE_ADDMM_CUDA_LT'] = '1' 
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
class EncodedVideoDataset(Dataset):
    """
    Dataset for loading encoded video latents.
    """
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]
        self.latent_files.sort()

        self.scaling_factor = 1.15
    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path, map_location='cpu').to(torch.float32)
        means, logvar = torch.split(latent,12,dim=1)
        # means = means * self.scaling_factor/ self.latents_std + self.latents_mean
        stds = (torch.randn_like(logvar) * torch.exp(0.5*logvar))

        out = means + stds
        out = out / self.scaling_factor
        return out#-out.mean()

# Example usage:
img_resolution = 128
latent_dir = f"encoded_latents/{img_resolution}x{img_resolution}"  # **Replace with your actual path**
batch_size = 16 
num_workers = 8 

dataset = EncodedVideoDataset(latent_dir)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True # Important for training stability
)

unet = UNet(img_resolution=img_resolution, # Match your latent resolution
            img_channels=12, # Match your latent channels
            label_dim = 0,
            model_channels=64,
            channel_mult=[1,2,4,8],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=1,
            )
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
sigma_data = 1.
precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to("cuda")
loss_fn = EDM2Loss(P_mean=-.4,P_std=1, sigma_data=sigma_data, noise_weight=MultiNoiseLoss())
loss_fn.noise_weight.loss_mean_popt =[0.2,0,1,0] 
loss_fn.noise_weight.loss_std_popt = [10,0.01,1e-4]

# this are vertical_scaling, x_min, width, vertical_offset, logistic_multiplier, logistic_offset
# Optimizermolto più appassionati e meno frustrati degli altri, a mio avviso perchè avevano anche delle loro aziende
logvar_params = [p for n, p in precond.named_parameters() if 'logvar' in n]
unet_params = unet.parameters()  # Get parameters from self.unet

optimizer = torch.optim.AdamW(precond.parameters(), lr=1e-3)

num_epochs = 500 # Adjust as needed

#%%
# torch.autograd.set_detect_anomaly(True, check_nan=True)
# Training loop
ulw=False
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):

        optimizer.zero_grad()
        batch = batch.to("cuda") #Scale down the latents

        # Calculate loss    
        loss = loss_fn(precond, batch, use_loss_weight=ulw)


        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    # Save model checkpoint (optional)
    if epoch % 3 == 0:
        loss_fn.noise_weight.plot('plot.png')
        loss_fn.noise_weight.fit_loss_curve()
    if epoch % (num_epochs // 5) == 0 and epoch!=0:  # save every 20% of epochs
        ulw=True
        torch.save({
            'epoch': epoch,
            'model_state_dict': precond.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"model_epoch_{epoch+1}.pt")


print("Training finished!")

# %%
def calculate_average_std_from_dataloader(dataloader):
    """
    Calculates the average standard deviation of encoded video latents directly from a DataLoader.

    Args:
        dataloader: The DataLoader providing batches of encoded latents.

    Returns:
        The average standard deviation across all latents in the dataset.
    """
    all_stds = []
    all_means = []
    for batch in dataloader:
      
        std = torch.std(batch, dim=(-1,-2,-3)).mean()
        all_stds.append(std.item())
        all_means.append(torch.mean(batch).item())

        

    average_std = sum(all_stds) / len(all_stds)
    average_mean = sum(all_means) / len(all_means)
    return average_mean,average_std

# %%
avg_mean_dataloader, avg_std_dataloader = calculate_average_std_from_dataloader(dataloader)
print(f"Average mean: {avg_mean_dataloader:.4f}, Average std: {avg_std_dataloader:.4f}")
# %%
