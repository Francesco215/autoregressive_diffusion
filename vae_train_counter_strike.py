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
from streaming import StreamingDataset


from edm2.gym_dataloader import GymDataGenerator, gym_collate_function
from edm2.dataloading import CsCollate, CsDataset
from edm2.utils import apply_clipped_grads
from edm2.vae import VAE, MixedDiscriminator
from colour_balanced_recon_loss import color_balanced_recon_loss

torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"

    batch_size = 8
    micro_batch_size = 2
    clip_length = 32 
    
    # Hyperparameters
    latent_channels = 8
    n_res_blocks = 2
    channels = [3, 32, 32, latent_channels]

    # Initialize models
    vae = VAE(channels = channels, n_res_blocks=n_res_blocks).to(device)
    # vae = VAE.load_from_pretrained('saved_models/vae_cs_4264.pt').to(device)
    # Example instantiation
    discriminator = MixedDiscriminator(in_channels = 3, block_out_channels=(32,)).to(device)
    
    dataset = CsDataset(clip_size=clip_length, remote='s3://counter-strike-data/original/', local = '/tmp/streaming_dataset/cs_vae',batch_size=micro_batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsCollate(clip_length), num_workers=8, shuffle=False)
    total_number_of_steps = len(dataloader)//micro_batch_size


    vae_params = sum(p.numel() for p in vae.parameters())
    discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of vae parameters: {vae_params//1e3}K")
    print(f"Number of discriminator parameters: {discriminator_params//1e3}K")
    # sigma_data = 0.434
    sigma_data = 1.

    # Define optimizers
    base_lr = 1e-4
    optimizer_vae = AdamW(vae.parameters(), lr=base_lr, eps=1e-8)
    optimizer_disc = AdamW(discriminator.parameters(), lr=base_lr, eps=1e-8)
    optimizer_vae.zero_grad()
    optimizer_disc.zero_grad()

    # Add exponential decay schedule
    gamma = 0.1 ** (1 / total_number_of_steps)  # Decay factor so lr becomes 0.1 * initial_lr after 40,000 steps
    scheduler_vae = lr_scheduler.ExponentialLR(optimizer_vae, gamma=gamma)
    scheduler_disc = lr_scheduler.ExponentialLR(optimizer_disc, gamma=gamma)
    losses = []


    recon_losses, kl_group_losses, kl_losses, disc_losses, adversarial_losses= [], [], [], [], []

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
            individual_var = logvar.exp().mean(dim=(0, 2, 3, 4))  # Average of individual variances
            mean_var = mean.var(dim=(0, 2, 3, 4))  # Variance of means
            group_var = individual_var + mean_var  # Total mixture variance
            group_mean = mean.mean(dim=(0, 2, 3, 4))
            kl_group = -0.5 * (1 + group_var.log() - group_mean.pow(2) - group_var).sum(dim=0)
            kl_loss  = -0.5 * (1 + logvar - logvar.exp()).sum(dim=1).mean()
            # kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1).mean()

            logits = discriminator(recon)
            targets = torch.ones(logits.shape[0], *logits.shape[2:], device=device, dtype=torch.long)

            adversarial_loss = F.cross_entropy(logits, targets)/np.log(2)

            # VAE losses
            recon_loss = F.l1_loss(recon, frames)

            # Define the loss components
            main_loss = recon_loss + kl_group*1e-3 + kl_loss*1e-3 + adversarial_loss*1e-2
            main_loss.backward()

            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()


            logits_real = discriminator(frames.detach())
            logits_fake = discriminator(recon.detach())
            loss_disc_real = F.cross_entropy(logits_real, torch.ones_like (targets))/np.log(2)
            loss_disc_fake = F.cross_entropy(logits_fake, torch.zeros_like(targets))/np.log(2)
            loss_disc = (loss_disc_real + loss_disc_fake)/2

            loss_disc.backward()
            # Update discriminator
            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
                optimizer_disc.step()
                scheduler_disc.step()
                optimizer_disc.zero_grad()
            
            pbar.set_postfix_str(f"recon loss: {recon_loss.item():.4f}, KL group loss: {kl_group.item():.4f} KL loss: {kl_loss.item():.4f}, Discr Loss: {loss_disc.item():.4f}, Adversarial Loss: {adversarial_loss.mean().item():.4f}")
            recon_losses.append(recon_loss.item())
            kl_group_losses.append(kl_group.item())
            kl_losses.append(kl_loss.item())
            adversarial_losses.append(adversarial_loss.mean().item())
            disc_losses.append(loss_disc.item())

            # if batch_idx == 500:
            #     adv_multiplier = 5e-2

            # Visualization every 100 steps
            if batch_idx % 1000 == 0 and batch_idx > 0:
                # Create a figure with a custom layout: 3 sections (2 rows for frames, 2x2 grid for losses)
                fig = plt.figure(figsize=(15, 12))

                # Top section: 2 rows for original and reconstructed frames (2x5 grid)
                gs_top = plt.GridSpec(2, 5, figure=fig, top=0.95, bottom=0.55, left=0.1, right=0.9)
                orig_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]  # Row 0: Original frames
                recon_axes = [fig.add_subplot(gs_top[1, i]) for i in range(5)]  # Row 1: Reconstructed frames

                # Bottom section: 2x2 grid for loss plots
                gs_bottom = plt.GridSpec(2, 2, figure=fig, top=0.45, bottom=0.05, left=0.1, right=0.9, hspace = 0.4)
                loss_axes = [
                    fig.add_subplot(gs_bottom[0, 0]),  # Top-left: Recon Loss
                    fig.add_subplot(gs_bottom[0, 1]),  # Top-right: KL Group Loss
                    fig.add_subplot(gs_bottom[1, 0]),  # Bottom-left: KL Loss
                    fig.add_subplot(gs_bottom[1, 1])   # Bottom-right: Disc Loss
                ]

                # --- Video Frames (Top: Original and Reconstructed) ---
                with torch.no_grad():
                    # Detach and denormalize frames and reconstructions
                    frames_denorm = (frames.cpu() + 1) / 2  # Shape: (batch, channels, time, height, width)
                    recon_denorm = (recon.cpu() + 1) / 2    # Shape: (batch, channels, time, height, width)

                    # Select the first sequence in the batch
                    frames_denorm = frames_denorm[0]  # Shape: (channels, time, height, width)
                    recon_denorm = recon_denorm[0]    # Shape: (channels, time, height, width)

                    # Clip values to [0, 1] to avoid numerical errors
                    frames_denorm = torch.clamp(frames_denorm, 0, 1)
                    recon_denorm = torch.clamp(recon_denorm, 0, 1)

                    # Rearrange to (time, height, width, channels) for plotting
                    frames_denorm = einops.rearrange(frames_denorm, 'c t h w -> t h w c')
                    recon_denorm = einops.rearrange(recon_denorm, 'c t h w -> t h w c')

                    # Select 5 frames evenly spaced (or fewer if sequence is short)
                    t = frames_denorm.shape[0]
                    num_frames_to_display = min(5, t)
                    indices = np.linspace(0, t-1, num_frames_to_display, dtype=int)

                    # Plot original frames (top row)
                    for i, idx in enumerate(indices):
                        orig_axes[i].imshow(frames_denorm[idx])
                        orig_axes[i].set_title(f"Orig t={idx}")
                        orig_axes[i].axis('off')

                    # Plot reconstructed frames (second row)
                    for i, idx in enumerate(indices):
                        recon_axes[i].imshow(recon_denorm[idx])
                        recon_axes[i].set_title(f"Recon t={idx}")
                        recon_axes[i].axis('off')
                    

                # --- Loss Plots (Bottom 2x2 Grid) ---
                # Titles and data for each loss plot
                loss_data = [recon_losses, kl_group_losses, kl_losses, disc_losses, adversarial_losses]
                loss_titles = ['Reconstruction Loss', 'KL Group Loss', 'KL Loss', 'Discriminator Losses', 'Discriminator Losses']
                loss_colors = ['blue', 'green', 'red', 'purple', 'orange']
                labels = [None, None, None, 'Discriminator Loss', 'Adversarial Loss']

                for i in range(5):
                    loss_axes[min(i,3)].plot(loss_data[i], color=loss_colors[i], label=labels[i])
                    loss_axes[min(i,3)].set_title(loss_titles[i])
                    loss_axes[min(i,3)].set_yscale('log')  
                    loss_axes[min(i,3)].set_xscale('log')  
                    loss_axes[min(i,3)].set_xlabel('Steps')
                    loss_axes[min(i,3)].set_xlim(left=10)
                    loss_axes[min(i,3)].set_ylabel('Loss')
                    loss_axes[min(i,3)].grid(True, linestyle='--', alpha=0.7)

                    
                    if i==4:
                        loss_axes[min(i,3)].legend()

                # Adjust layout to fit everything nicely
                plt.tight_layout()

                # Save the combined plot
                os.makedirs("images_training", exist_ok=True)
                plt.savefig(f"images_training/combined_step_cs_{batch_idx}.png")
                plt.close()
            if batch_idx % (total_number_of_steps//10) == 0 and batch_idx != 0:
                os.makedirs("saved_models", exist_ok=True)
                vae.save_to_state_dict(f'saved_models/vae_cs_{batch_idx}.pt')

    print("Finished Training")
    # %%