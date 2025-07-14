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

# Import LPIPS
import lpips 

from edm2.cs_dataloading import CsCollate, CsDataset
from edm2.vae import VAE, MixedDiscriminator
from edm2.utils import GaussianLoss

# torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = "cuda"

    batch_size = 4
    micro_batch_size = 4
    clip_length = 32

    # Hyperparameters
    latent_channels = 8
    n_res_blocks = 3
    channels = [3, 16, 64, 256, latent_channels]

    # Initialize models
    vae = VAE(channels = channels, n_res_blocks=n_res_blocks, spatial_compressions=[1,2,2,2], time_compressions=[1,2,2,1]).to(device)
    # vae = VAE.from_pretrained('saved_models/vae_cs_15990.pt').to(device)
    vae = torch.compile(vae)
    #%%

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

    # --- Scheduler Definition ---
    warmup_steps = 100
    decay_factor = 0.1 # The factor by which the LR will be decayed

    # Calculate gamma for the exponential decay part of the schedule
    # This ensures the LR decays to `decay_factor` * `base_lr` over the steps following the warmup
    gamma = decay_factor ** (1 / (total_number_of_steps - warmup_steps))

    # Define the learning rate schedule function
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # The decay starts after the warmup is complete
            return gamma ** (current_step - warmup_steps)

    # Add the combined warmup and decay schedule
    scheduler_vae = lr_scheduler.LambdaLR(optimizer_vae, lr_lambda)

    # Initialize LPIPS loss function <--- ADDED
    lpips_loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_loss_fn.cuda()

    # Store losses
    gaussian_recon_losses, l1_recon_losses, lpips_losses = [], [], [] # <--- MODIFIED

    #%%
    # Training loop
    for _ in range(10):
        pbar = tqdm(enumerate(dataloader), total=total_number_of_steps)
        for batch_idx, micro_batch in pbar:
            with torch.no_grad():
                frames, _ = micro_batch # Ignore actions and reward for this VggAE training
                frames = frames.float() / 127.5 - 1 # Normalize to [-1, 1]
                frames = einops.rearrange(frames, 'b t h w c-> b c t h w').to(device)

            # VAE forward pass
            # r_mean (reconstruction mean): This is your hat_x
            # r_logvar (reconstruction log variance): Used for GaussianLoss
            # mean (latent mean), logvar (latent log variance): For KL divergence
            r_mean, r_logvar, mean, _ = vae(frames)

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

            main_loss = gaussian_loss + lpips_weight * lpips_loss
            main_loss.backward()

            if batch_idx % (batch_size//micro_batch_size) == 0 and batch_idx!=0:
                nn.utils.clip_grad_norm_(vae.parameters(), 0.5)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()

            # Update tqdm progress bar <--- MODIFIED
            pbar.set_postfix_str(f"gaussian_recon: {gaussian_loss.item():.4f}, l1_recon: {l1_loss.item():.4f}, lpips: {lpips_loss.item():.4f}, current_lr: {optimizer_vae.param_groups[0]['lr']:.4f}")
            gaussian_recon_losses.append(gaussian_loss.item()) # Store all loss components for plotting
            l1_recon_losses.append(l1_loss.item())
            lpips_losses.append(lpips_loss.item()) # <--- ADDED


            if batch_idx % 1000 == 0 and batch_idx > 0:
                fig = plt.figure(figsize=(15, 18)) # <--- Increased figure height for the new row
                fig.suptitle(f"VAE Training Progress - VAE Parameters: {vae_params//1e6}M", fontsize=16)
                # Top section: 3 rows for original, reconstructed (mean), and uncertainty heatmaps
                gs_top = plt.GridSpec(3, 5, figure=fig, top=0.95, bottom=0.5, left=0.1, right=0.9) # <--- Adjusted bottom margin
                orig_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
                recon_mean_axes = [fig.add_subplot(gs_top[1, i]) for i in range(5)] # New row for mean
                uncertainty_axes = [fig.add_subplot(gs_top[2, i]) for i in range(5)] # New row for uncertainty

                # Bottom section: 1x3 for loss plots <--- MODIFIED (was 1x2)
                gs_bottom = plt.GridSpec(1, 3, figure=fig, top=0.45, bottom=0.1, left=0.05, right=0.95, hspace=0.4, wspace=0.3) # <--- Adjusted wspace
                loss_axes = [
                    fig.add_subplot(gs_bottom[0, 0]),
                    fig.add_subplot(gs_bottom[0, 1]),
                    fig.add_subplot(gs_bottom[0, 2]),
                ]

                # Frame visualization
                with torch.no_grad():
                    frames_denorm = (frames.cpu() + 1) / 2
                    recon_mean_denorm = (r_mean.cpu() + 1) / 2 # Plotting the mean directly
                    
                    # Calculate uncertainty (variance)
                    uncertainty = torch.exp(r_logvar).cpu() # Variance
                    
                    # We take the mean across the channel dimension for visualization
                    uncertainty_to_plot = torch.mean(uncertainty, dim=1, keepdim=True) # Mean over channels
                    
                    # Ensure all tensors are clamped to [0, 1] for plotting, except for uncertainty which should reflect its true range
                    frames_denorm = torch.clamp(frames_denorm[0], 0, 1) # (c, t, h, w)
                    recon_mean_denorm = torch.clamp(recon_mean_denorm[0], 0, 1) # (c, t, h, w)
                    uncertainty_to_plot = uncertainty_to_plot[0] # Select the first batch item (1, t, h, w)


                    frames_denorm = einops.rearrange(frames_denorm, 'c t h w -> t h w c')
                    recon_mean_denorm = einops.rearrange(recon_mean_denorm, 'c t h w -> t h w c')
                    uncertainty_to_plot = einops.rearrange(uncertainty_to_plot, 'c t h w -> t h w c') # Now (t, h, w, 1)
                    
                    t_idx = frames_denorm.shape[0]
                    indices = np.linspace(0, t_idx - 1, 5, dtype=int)

                    # Calculate global min/max for uncertainty across the displayed batch for consistent colorbar
                    global_min_uncertainty = uncertainty_to_plot.min().item()
                    global_max_uncertainty = uncertainty_to_plot.max().item()

                    for i, idx in enumerate(indices):
                        # Original Frames
                        orig_axes[i].imshow(frames_denorm[idx])
                        orig_axes[i].set_title(f"Orig t={idx}")
                        orig_axes[i].axis('off')

                        # Reconstructed Mean
                        recon_mean_axes[i].imshow(recon_mean_denorm[idx])
                        recon_mean_axes[i].set_title(f"Recon Mean t={idx}")
                        recon_mean_axes[i].axis('off')

                        # Uncertainty Heatmap
                        uncertainty_axes[i].imshow(recon_mean_denorm[idx]) # Display the mean image
                        # Overlay heatmap. Squeeze the channel dimension for imshow.
                        im = uncertainty_axes[i].imshow(uncertainty_to_plot[idx].squeeze(-1), cmap='viridis', alpha=0.6,
                                                         vmin=global_min_uncertainty, vmax=global_max_uncertainty)
                        uncertainty_axes[i].set_title(f"Uncertainty Heatmap t={idx}")
                        uncertainty_axes[i].axis('off')
                        # Add a colorbar for the first uncertainty plot
                        if i == 0:
                            fig.colorbar(im, ax=uncertainty_axes[i], orientation='vertical', fraction=0.046, pad=0.04)


                # Plot Gaussian Reconstruction loss (formerly "Recon Loss")
                loss_axes[0].plot(gaussian_recon_losses, label="Gaussian Loss", color="orange") # Plot gaussian losses
                loss_axes[0].set_title("Gaussian Losses") # Modified title to reflect both
                loss_axes[0].set_xscale("log")
                loss_axes[0].set_xlabel("Steps")
                loss_axes[0].set_ylabel("Loss")
                # loss_axes[0].legend() # Add legend
                loss_axes[0].grid(True)

                loss_axes[1].plot(l1_recon_losses, label="L1 Recon Loss", color="blue") # Plot L1 Loss
                loss_axes[1].set_title("L1 Reconstruction Losses") # Modified title to reflect both
                loss_axes[1].set_yscale("log")
                loss_axes[1].set_xscale("log")
                loss_axes[1].set_xlabel("Steps")
                loss_axes[1].set_ylabel("Loss")
                # loss_axes[1].legend() # Add legend
                loss_axes[1].grid(True)

                # Plot LPIPS loss <--- ADDED NEW PLOT
                loss_axes[2].plot(lpips_losses, label="LPIPS Loss", color="green")
                loss_axes[2].set_title("LPIPS Loss")
                loss_axes[2].set_yscale("log")
                loss_axes[2].set_xscale("log")
                loss_axes[2].set_xlabel("Steps")
                loss_axes[2].set_ylabel("Loss")
                loss_axes[2].grid(True)

                plt.tight_layout()
                os.makedirs("images_training", exist_ok=True)
                plt.savefig(f"images_training/combined_step_cs_{batch_idx}.png")
                plt.close()

            if batch_idx % (total_number_of_steps // 10) == 0 and batch_idx != 0:
                os.makedirs("saved_models", exist_ok=True)
                vae.save_to_state_dict(f'saved_models/vae_cs_{batch_idx}.pt')

    print("Finished Training")
    #%%