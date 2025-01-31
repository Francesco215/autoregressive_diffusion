#%%
import gc
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import IterableDataset
from matplotlib import pyplot as plt

from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.loss_weight import MultiNoiseLoss
from edm2.mars import MARS
from edm2.dataloading import OpenVidDataloader, RandomDataset, OpenVidDataset
from edm2.phema import PowerFunctionEMA
from edm2.sampler import edm_sampler

# import logging
# torch._logging.set_logs(dynamo=logging.INFO)

torch._dynamo.config.recompile_limit = 100
# Example usage:
n_clips = 100_000
micro_batch_size = 4 
batch_size = 32
accumulation_steps = batch_size//micro_batch_size
total_number_of_batches = n_clips // batch_size
total_number_of_steps = total_number_of_batches * accumulation_steps


num_workers = 8 
device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = OpenVidDataloader(micro_batch_size, num_workers, device, dataset = OpenVidDataset())
# dataloader = RandomDataloader(micro_batch_size, num_workers, device)


unet = UNet(img_resolution=64, # Match your latent resolution
            img_channels=16, # Match your latent channels
            label_dim = dataloader.text_embedding_dim,
            model_channels=128,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=2,
            attn_resolutions=[16,8]
            )
unet_params = sum(p.numel() for p in unet.parameters())//1e6
print(f"Number of UNet parameters: {unet_params}M")
# sigma_data = 0.434
sigma_data = 1.
precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to("cuda")
loss_fn = EDM2Loss(P_mean=0.5,P_std=1.5, sigma_data=sigma_data, noise_weight=MultiNoiseLoss())

ref_lr = 1e-2
current_lr = ref_lr
optimizer = MARS(precond.parameters(), lr=ref_lr, eps = 1e-4)
optimizer.zero_grad()

ema_tracker = PowerFunctionEMA(precond, stds=[0.050, 0.100])

#%%
# torch.autograd.set_detect_anomaly(True, check_nan=True)
# Training loop
ulw=False
losses = []
pbar = tqdm(enumerate(dataloader),total=total_number_of_steps)
for i, micro_batch in pbar:
    if i==0: print("Downloaded first batch and starting training loop")
    latents = micro_batch['latents'].to(device)
    text_embeddings = None if i%3==0 else micro_batch['text_embeddings'].to(device)

    # Calculate loss    
    loss, un_weighted_loss = loss_fn(precond, latents, text_embeddings, use_loss_weight=ulw)
    losses.append(un_weighted_loss)
    # Backpropagation and optimization
    loss.backward()
    pbar.set_postfix_str(f"Loss: {np.mean(losses[-accumulation_steps:]):.4f}, lr: {current_lr:.4f}")

    if i % accumulation_steps == 0 and i!=0:
        #microbatching
        optimizer.step()
        optimizer.zero_grad()
        ema_tracker.update(cur_nimg= i * batch_size, batch_size=batch_size)

        for g in optimizer.param_groups:
            current_lr = learning_rate_schedule(i, ref_lr, total_number_of_steps/50, 0)
            g['lr'] = current_lr

    # Save model checkpoint (optional)
    if i % 50 * accumulation_steps == 0 and i!=0:
        loss_fn.noise_weight.fit_loss_curve()
        loss_fn.noise_weight.plot('plot.png')
        n_clips = np.linspace(0, i * micro_batch_size, len(losses))
        plt.plot(n_clips, losses, label='Loss', color='blue', alpha=0.5)
        if len(losses) >= 100:
            moving_avg = np.convolve(losses, np.ones(100) / 100, mode='valid')
            n_images_avg = np.linspace(0, i * micro_batch_size, len(moving_avg))
            plt.plot(n_images_avg, moving_avg, label='Moving Average', color='blue', alpha=1)
        plt.xscale('log')
        plt.xlabel('n images')
        plt.ylabel('loss')
        plt.title(f'Losses with {unet_params} parameters')
        plt.savefig('losses.png')
        plt.show()
        plt.close()
        ulw=True


    if i % (total_number_of_steps//4) == 0 and i!=0:  # save every 10% of epochs
        torch.save({
            'batch': i,
            'model_state_dict': precond.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'ema_state_dict': ema_tracker.state_dict(),
            'loss': loss,
        }, f"model_batch_{i}.pt")

    if i == total_number_of_steps:
        break

print("Training finished!")

# %%

import torch
from edm2.sampler import edm_sampler

precond.load_state_dict(torch.load("model_batch_5001.pt", weights_only=False)['model_state_dict'])

precond.eval()
#%%
samples = next(iter(dataloader))
# %%
x = samples["latents"][0:4,:-10].to("cuda")
for _ in range(10):
    y=edm_sampler(precond, x)
    print(x.shape,y.shape)
    y[:,:-1]=x
    x=y.clone()


# %%
torch.save(y, "sampled_latents.pt")
# %%