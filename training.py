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
micro_batch_size = 12 
accumulation_steps = 8
batch_size = micro_batch_size * accumulation_steps

total_number_of_batches = 1000
num_workers = 32 
device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = OpenVidDataloader(micro_batch_size, num_workers, device)


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
optimizer.zero_grad()


#%%
# torch.autograd.set_detect_anomaly(True, check_nan=True)
# Training loop
ulw=False
total_number_of_steps = total_number_of_batches * accumulation_steps
for i, micro_batch in tqdm(enumerate(dataloader),total=total_number_of_steps):
    latents = micro_batch['latents'].to(device)

    # Calculate loss    
    loss = loss_fn(precond, latents, use_loss_weight=ulw)


    # Backpropagation and optimization
    loss.backward()
    if i % accumulation_steps == 0:
        #microbatching
        optimizer.step()
        optimizer.zero_grad()
        tqdm.write(f"Batch: {i}, Loss: {loss.item():.4f}")

    # Save model checkpoint (optional)
    if i % 50 * accumulation_steps == 0:
        loss_fn.noise_weight.plot('plot.png')
        loss_fn.noise_weight.fit_loss_curve()
    if i % (total_number_of_steps//5) == 0 and i!=0:  # save every 20% of epochs
        torch.save({
            'batch': i,
            'model_state_dict': precond.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"model_batch_{i+1}.pt")
    if i == total_number_of_steps:
        break

print("Training finished!")

# %%

import torch
from edm2.sampler import edm_sampler

# %%
precond.eval()
x = next(iter(dataloader)).to("cuda")[0,:-10]
# %%
for _ in range(10):
    y=edm_sampler(precond, x)
    print(x.shape,y.shape)
    y[:,:-1]=x
    x=y.clone()


# %%
torch.save(y, "sampled_latents.pt")