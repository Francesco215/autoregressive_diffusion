#%%
import os
import einops
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

from edm2.networks_edm2 import UNet, Precond
from edm2.training_loop import EDM2Loss

class EncodedVideoDataset(Dataset):
    """
    Dataset for loading encoded video latents.
    """
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]
        self.latent_files.sort()
        self.scaling_factor = 1.15258426 # Not used currently

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path, map_location='cpu')
        latent = einops.rearrange(latent, 't c h w -> c t h w')
        return latent

# Example usage:
img_resolution = 128
latent_dir = f"encoded_latents/{img_resolution}x{img_resolution}"  # **Replace with your actual path**
batch_size = 32 
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
            img_channels=24,
            label_dim = 0,
            model_channels=64,
            channel_mult=[1,2,4,8],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=3,
            )
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
precond = Precond(unet, use_fp16=True, sigma_data=1., logvar_channels=128).to("cuda")
loss_fn = EDM2Loss(sigma_data=1.)

# Optimizer
logvar_params = [p for n, p in precond.named_parameters() if 'logvar' in n]
unet_params = unet.parameters()  # Get parameters from self.unet

optimizer = torch.optim.AdamW([
    {'params': unet_params, 'lr': 1e-3},
    {'params': logvar_params, 'lr': 1e-1}
])

num_epochs = 1000 # Adjust as needed

#%%
# torch.autograd.set_detect_anomaly(True, check_nan=True)
# Training loop
for epoch in range(num_epochs):
    if epoch%2 == 0:
        c_noise = torch.linspace(0.1, 80, 128, device="cuda")
        logvars = precond.logvar_linear(precond.logvar_fourier(c_noise)).flatten().cpu().detach().numpy()

        plt.close()
        plt.plot(c_noise.cpu().numpy(), logvars)
        plt.savefig(f"logvars_epoch.png")
        plt.show()   
    for i, batch in enumerate(dataloader):

        optimizer.zero_grad()
        batch = batch.to("cuda") / 3 # Scale down the latents

        # Calculate loss
        loss = loss_fn(precond, batch)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()


    print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")


    # Save model checkpoint (optional)
    if (epoch + 1) % num_epochs//5 == 0:  # save every 20% of epochs
         torch.save({
            'epoch': epoch,
            'model_state_dict': precond.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"model_epoch_{epoch+1}.pt")

print("Training finished!")
# %%
