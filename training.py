#%%
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt

from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss
from edm2.loss_weight import MultiNoiseLoss
from edm2.mars import MARS
from edm2.dataloading import OpenVidDataloader


# import logging
# torch._logging.set_logs(dynamo=logging.INFO)

# Example usage:
batch_size = 12 
num_workers = 32 
device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = OpenVidDataloader(batch_size, num_workers, device)


unet = UNet(img_resolution=64, # Match your latent resolution
            img_channels=16, # Match your latent channels
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

optimizer = MARS(precond.parameters(), lr=1e-4, eps = 1e-4)

num_epochs = 50 # Adjust as needed

#%%
# torch.autograd.set_detect_anomaly(True, check_nan=True)
# Training loop
ulw=False
for i, batch in tqdm(enumerate(dataloader)):
    latents = batch['latents']
    optimizer.zero_grad()
    latents = latents.to("cuda") 

    # Calculate loss    
    loss = loss_fn(precond, latents, use_loss_weight=ulw)


    # Backpropagation and optimization
    loss.backward()
    optimizer.step()
    tqdm.write(f"Batch: {i}, Loss: {loss.item():.4f}")

    # Save model checkpoint (optional)
    if i % 50 == 0:
        loss_fn.noise_weight.plot('plot.png')
        loss_fn.noise_weight.fit_loss_curve()
    # if i % 500 and i!=0:  # save every 20% of epochs
    #     torch.save({
    #         'batch': i,
    #         'model_state_dict': precond.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #     }, f"model_batch_{i+1}.pt")


print("Training finished!")

# %%

import torch
from edm2.sampler import edm_sampler

# %%
precond.eval()
x = next(iter(dataloader)).to("cuda")[:,:-10]
# %%
for _ in range(10):
    y=edm_sampler(precond, x)
    print(x.shape,y.shape)
    y[:,:-1]=x
    x=y.clone()


# %%
torch.save(y, "sampled_latents.pt")

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
