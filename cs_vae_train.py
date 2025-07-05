#%%
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from edm2.cs_dataloading import CsCollate, CsDataset
from edm2.vae import VAE, MixedDiscriminator

torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"

    batch_size = 8
    micro_batch_size = 2
    clip_length = 24 
    
    # Hyperparameters
    latent_channels = 8
    n_res_blocks = 2
    channels = [3, 64, 64, 64, latent_channels]

    # Initialize models
    vae = VAE(channels = channels, n_res_blocks=n_res_blocks, spatial_compressions=[1,2,2,2], time_compressions=[1,2,2,1]).to(device)
    # vae = VAE.load_from_pretrained('saved_models/vae_cs_4264.pt').to(device)
    # Example instantiation
    
    dataset = CsDataset(clip_size=clip_length, remote='s3://counter-strike-data/original/', local = '/tmp/streaming_dataset/cs_vae',batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsCollate(clip_length), num_workers=8, shuffle=False)
    total_number_of_steps = len(dataloader)//micro_batch_size


    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"Number of vae parameters: {vae_params//1e3}K")
    # sigma_data = 0.434
    sigma_data = 1.

    # Define optimizers
    base_lr = 1e-4
    optimizer_vae = AdamW(vae.parameters(), lr=base_lr, eps=1e-8)
    optimizer_vae.zero_grad()

    # Add exponential decay schedule
    gamma = 0.1 ** (1 / total_number_of_steps)  # Decay factor so lr becomes 0.1 * initial_lr after 40,000 steps
    scheduler_vae = lr_scheduler.ExponentialLR(optimizer_vae, gamma=gamma)
    losses = []


    recon_losses, kl_losses = [], []

    #%%
    # Training loop
    for _ in range(10):
        pbar = tqdm(enumerate(dataloader), total=total_number_of_steps)
        for batch_idx, micro_batch in pbar:
            with torch.no_grad():
                frames, _ = micro_batch  # Ignore actions and reward for this VggAE training
                frames = frames.float() / 127.5 - 1  # Normalize to [-1, 1]
                frames = einops.rearrange(frames, 'b t h w c-> b c t h w').to(device)

            # VAE forward pass
            recon, mean, logvar, _ = vae(frames)

            # in theory the mean should be only with respect to the batch size
            kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1).mean()

            # VAE losses
            recon_loss = F.l1_loss(recon, frames)

            # Define the loss components
            main_loss = recon_loss + kl_loss*1e-3 
            main_loss.backward()

            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()


            pbar.set_postfix_str(f"recon loss: {recon_loss.item():.4f}, KL loss: {kl_loss.item():.4f}")
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

            # if batch_idx == 500:
            #     adv_multiplier = 5e-2

            # Visualization every 100 steps

            
            if batch_idx % 10 == 0 and batch_idx > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                axes[0].plot(recon_losses, label="Recon Loss", color="blue")
                axes[0].set_title("Reconstruction Loss")
                axes[0].set_yscale("log")
                axes[0].set_xscale("log")
                axes[0].set_xlabel("Steps")
                axes[0].set_ylabel("Loss")
                axes[0].grid(True)

                axes[1].plot(kl_losses, label="KL Loss", color="red")
                axes[1].set_title("KL Loss")
                axes[1].set_yscale("log")
                axes[1].set_xscale("log")
                axes[1].set_xlabel("Steps")
                axes[1].set_ylabel("Loss")
                axes[1].grid(True)

                plt.tight_layout()
                os.makedirs("images_training", exist_ok=True)
                plt.savefig(f"images_training/loss_plot_step_{batch_idx}.png")
                plt.close()

            if batch_idx % (total_number_of_steps // 10) == 0 and batch_idx != 0:
                os.makedirs("saved_models", exist_ok=True)
                vae.save_to_state_dict(f'saved_models/vae_cs_{batch_idx}.pt')

    print("Finished Training")
    # %%