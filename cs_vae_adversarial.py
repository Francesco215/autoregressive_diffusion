#%%
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW

import os
import lpips
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from streaming.base.util import clean_stale_shared_memory


from edm2.cs_dataloading import CsCollate, CsDataset
from edm2.utils import GaussianLoss
from edm2.vae import VAE, MixedDiscriminator

torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    clean_stale_shared_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"

    batch_size = 8
    micro_batch_size = 4
    clip_length = 32 

    # Initialize models
    vae = VAE.from_pretrained('saved_models/vae_cs_10660.pt').requires_grad_(False).to(device)
    vae.decoder.encoder_blocks[-1].requires_grad_(True)
    # Example instantiation
    discriminator = MixedDiscriminator(in_channels = 3, block_out_channels=(32,)).to(device)
    
    dataset = CsDataset(clip_size=clip_length, remote='s3://counter-strike-data/original/', local = '/tmp/streaming_dataset/cs_vae',batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
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
    optimizer_vae = AdamW((p for p in vae.parameters() if p.requires_grad), lr=base_lr, eps=1e-8)
    optimizer_disc = AdamW(discriminator.parameters(), lr=base_lr, eps=1e-8)
    optimizer_vae.zero_grad()
    optimizer_disc.zero_grad()

    # Add exponential decay schedule
    gamma = 0.1 ** (1 / total_number_of_steps)  # Decay factor so lr becomes 0.1 * initial_lr after 40,000 steps
    scheduler_vae = lr_scheduler.ExponentialLR(optimizer_vae, gamma=gamma)
    scheduler_disc = lr_scheduler.ExponentialLR(optimizer_disc, gamma=gamma)
    losses = []
    lpips_loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_loss_fn.cuda()


    recon_losses, disc_losses, adversarial_losses= [], [], []

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
            # r_mean (reconstruction mean): This is your hat_x
            # r_logvar (reconstruction log variance): Used for GaussianLoss
            # mean (latent mean), logvar (latent log variance): For KL divergence
            r_mean, r_logvar, mean, logvar, _ = vae(frames)

            # in theory the mean should be only with respect to the batch size
            kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1).mean()

            # VAE losses
            gaussian_loss = GaussianLoss(r_mean, r_logvar, frames)
            l1_loss = F.l1_loss(r_mean, frames)

            # --- MODIFIED LPIPS Calculation ---
            # Reshape frames and r_mean from [B, C, T, H, W] to [B*T, C, H, W]
            # so that LPIPS can process them as individual images.
            # Your frames are C=3, and T=32, so we need to flatten T into the batch dimension.
            frames_flat = torch.clip(einops.rearrange(frames, 'b c t h w -> (b t) c h w'), -1, 1)
            r_mean_flat = torch.clip(einops.rearrange(r_mean, 'b c t h w -> (b t) c h w'), -1, 1)

            # Calculate LPIPS loss for each frame and then take the mean
            lpips_loss = lpips_loss_fn(r_mean_flat, frames_flat).mean()
            # --- END MODIFIED LPIPS Calculation ---


            # Define the loss components
            lpips_weight = 0.1
            kl_weight = 1e-4

            main_loss = gaussian_loss + kl_weight * kl_loss + lpips_weight * lpips_loss
            logits = discriminator(r_mean)
            targets = torch.ones(logits.shape[0], *logits.shape[2:], device=device, dtype=torch.long)

            adversarial_loss = F.cross_entropy(logits, targets)/np.log(2)

            # Define the loss components
            loss = gaussian_loss + kl_weight * kl_loss + lpips_weight * lpips_loss + adversarial_loss*1e-1
            loss.backward()


            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()


            logits_real = discriminator(frames.detach())
            logits_fake = discriminator(r_mean.detach())
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
            
            pbar.set_postfix_str(f"recon loss: {l1_loss.item():.4f}, Discr Loss: {loss_disc.item():.4f}, Adversarial Loss: {adversarial_loss.mean().item():.4f}")
            recon_losses.append(l1_loss.item())
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
                gs_bottom = plt.GridSpec(1, 2, figure=fig, top=0.45, bottom=0.05, left=0.1, right=0.9, hspace = 0.4)
                loss_axes = [
                    fig.add_subplot(gs_bottom[0, 0]),  # Top-left: Recon Loss
                    fig.add_subplot(gs_bottom[0, 1]),  # Top-right: KL Group Loss
                ]

                # --- Video Frames (Top: Original and Reconstructed) ---
                with torch.no_grad():
                    # Detach and denormalize frames and reconstructions
                    frames_denorm = (frames.cpu() + 1) / 2  # Shape: (batch, channels, time, height, width)
                    recon_denorm = (r_mean.cpu() + 1) / 2    # Shape: (batch, channels, time, height, width)

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
                loss_data = [recon_losses, disc_losses, adversarial_losses]
                loss_titles = ['Reconstruction Loss', 'Discriminator Losses', 'Discriminator Losses']
                loss_colors = ['blue', 'red', 'orange']
                labels = [None, 'Discriminator Loss', 'Adversarial Loss']

                for i in range(len(loss_data)):
                    loss_axes[min(i,1)].plot(loss_data[i], color=loss_colors[i], label=labels[i])
                    loss_axes[min(i,1)].set_title(loss_titles[i])
                    loss_axes[min(i,1)].set_yscale('log')  
                    loss_axes[min(i,1)].set_xscale('log')  
                    loss_axes[min(i,1)].set_xlabel('Steps')
                    loss_axes[min(i,1)].set_xlim(left=10)
                    loss_axes[min(i,1)].set_ylabel('Loss')
                    loss_axes[min(i,1)].grid(True, linestyle='--', alpha=0.7)

                    
                    if i==2:
                        loss_axes[min(i,1)].legend()

                # Adjust layout to fit everything nicely
                plt.tight_layout()

                # Save the combined plot
                os.makedirs("images_training", exist_ok=True)
                plt.savefig(f"images_training/adversarial_cs_{batch_idx}.png")
                plt.close()
            if batch_idx % (total_number_of_steps//10) == 0 and batch_idx != 0:
                os.makedirs("saved_models", exist_ok=True)
                vae.save_to_state_dict(f'saved_models/vae_cs_adv{batch_idx}.pt')

    print("Finished Training")
    # %%