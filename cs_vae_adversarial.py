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
from streaming.base.util import clean_stale_shared_memory


from edm2.cs_dataloading import CsCollate, CsDataset
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
    vae = VAE.from_pretrained('saved_models/vae_cs_62040.pt').requires_grad_(False).to(device)
    vae.decoder.encoder_blocks[-2].requires_grad_(True)
    vae.decoder.encoder_blocks[-1].requires_grad_(True)
    vae = torch.compile(vae)
    # Example instantiation
    discriminator = MixedDiscriminator(in_channels = 3, block_out_channels=(64,32)).to(device)
    
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


    recon_losses, disc_losses, adversarial_losses= [], [], []

    #%%
    # Training loop
  
    # --- Training loop ---
    for epoch in range(10):
        pbar = tqdm(dataloader)
        for batch_idx, micro_batch in enumerate(pbar):
            frames, _ = micro_batch
            frames = frames.float() / 127.5 - 1
            frames = einops.rearrange(frames, 'b t h w c -> b c t h w').to(device)

            # ===================================
            #       TRAIN THE VAE (GENERATOR)
            # ===================================
            optimizer_vae.zero_grad()

            # VAE forward pass to get reconstructed frames
            recon, _, _, _ = vae(frames)

            # 1. Reconstruction Loss (e.g., L1 loss)
            recon_loss = F.l1_loss(recon, frames)

            # 2. Adversarial Loss for the VAE
            # We want the VAE to fool the discriminator.
            # DO NOT detach `recon` here, so gradients can flow to the VAE.
            with torch.no_grad():
                logits_real_for_vae = discriminator(frames) # Detach since we don't need grads wrt D
            logits_fake_for_vae = discriminator(recon)
        
            # Generator wants to minimize the difference, which is the negative of the discriminator's objective
            vae_adv_loss = -torch.log(1 + torch.exp(logits_real_for_vae - logits_fake_for_vae)).mean()
        
            # Total VAE loss
            vae_total_loss = recon_loss + vae_adv_loss * 0.1 # You might want to weight the adv_loss

            # Backpropagate and update the VAE
            vae_total_loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(vae.parameters(), 1) # Clipping can still be useful
            optimizer_vae.step()
            scheduler_vae.step()

            # ===================================
            #      TRAIN THE DISCRIMINATOR
            # ===================================
            optimizer_disc.zero_grad()
        
            # We can reuse `recon` from the VAE step, but we MUST detach it now.
            recon_detached = recon.detach()
        
            # 1. Adversarial Loss for the Discriminator
            logits_real = discriminator(frames)
            logits_fake = discriminator(recon_detached)
        
            # Discriminator wants to maximize the difference
            disc_adv_loss = torch.log(1 + torch.exp(logits_real - logits_fake)).mean()

            # Total discriminator loss
            disc_total_loss = disc_adv_loss 
        
            # Backpropagate and update the discriminator
            disc_total_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            optimizer_disc.step()
            scheduler_disc.step()
        
            pbar.set_postfix_str(f"Epoch {epoch+1}, VAE Loss: {recon_loss.item():.4f}, Disc Loss: {disc_total_loss.item():.4f}")

            recon_losses.append(recon_loss.item())
            disc_losses.append(disc_total_loss.item())
            adversarial_losses.append(vae_adv_loss.item())
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