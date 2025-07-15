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

# torch.autograd.set_detect_anomaly(True)
torch._dynamo.config.recompile_limit = 100
if __name__=="__main__":
    clean_stale_shared_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"

    batch_size = 4
    micro_batch_size = 2
    clip_length = 32 

    # Initialize models
    vae = VAE.from_pretrained('saved_models/vae_cs_23452.pt').requires_grad_(False).to(device)
    vae.decoder.encoder_blocks[-1:].requires_grad_(True)
    vae=torch.compile(vae)
    # Example instantiation
    discriminator = MixedDiscriminator(in_channels = 6, block_out_channels=(64,128,64)).to(device)
    discriminator = torch.compile(discriminator)
    
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
    warmup_steps = 1000
    decay_factor = 0.4 # The factor by which the LR will be decayed

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
    # scheduler_disc = lr_scheduler.ExponentialLR(optimizer_disc, gamma=gamma)
    losses = []
    lpips_loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_loss_fn.cuda()


    recon_losses, disc_losses, adversarial_losses, gaussian_recon_losses, lpips_losses= [], [], [], [], []

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

            adversarial_loss = discriminator.vae_loss(frames,r_mean)
            # Define the loss components
            loss = gaussian_loss + lpips_loss*1e-1 + adversarial_loss *0.4
            loss.backward()


            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()


            loss_disc = discriminator.discriminator_loss(frames, r_mean)
            loss_disc.backward()
            # Update discriminator
            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
                optimizer_disc.step()
                # scheduler_disc.step()
                optimizer_disc.zero_grad()
            
            pbar.set_postfix_str(f"recon loss: {l1_loss.item():.4f}, Discr Loss: {loss_disc.item():.4f}, Adversarial Loss: {adversarial_loss.mean().item():.4f}")
            recon_losses.append(l1_loss.item())
            adversarial_losses.append(adversarial_loss.mean().item())
            disc_losses.append(loss_disc.item())
            gaussian_recon_losses.append(gaussian_loss.item()) # Store all loss components for plotting
            lpips_losses.append(lpips_loss.item()) # <--- ADDED

            # if batch_idx == 500:
            #     adv_multiplier = 5e-2

            # Visualization every 100 steps
            if batch_idx % 100 == 0 and batch_idx > 0:
                fig = plt.figure(figsize=(15, 18)) # <--- Increased figure height for the new row
                fig.suptitle(f"VAE Training Progress - VAE Parameters: {vae_params//1e6}M", fontsize=16)
                # Top section: 3 rows for original, reconstructed (mean), and uncertainty heatmaps
                gs_top = plt.GridSpec(3, 5, figure=fig, top=0.95, bottom=0.5, left=0.1, right=0.9) # <--- Adjusted bottom margin
                orig_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
                recon_mean_axes = [fig.add_subplot(gs_top[1, i]) for i in range(5)] # New row for mean
                uncertainty_axes = [fig.add_subplot(gs_top[2, i]) for i in range(5)] # New row for uncertainty

                # Bottom section: 2x2 grid for loss plots
                gs_bottom = plt.GridSpec(2, 2, figure=fig, top=0.45, bottom=0.05, left=0.1, right=0.9, hspace = 0.4)
                loss_axes = [
                    fig.add_subplot(gs_bottom[0, 0]),  # Top-left: Recon Loss
                    fig.add_subplot(gs_bottom[0, 1]),  # Top-right: KL Group Loss
                    fig.add_subplot(gs_bottom[1, 0]),  # Top-right: KL Group Loss
                    fig.add_subplot(gs_bottom[1, 1]),  # Top-right: KL Group Loss
                ]

                # --- Video Frames (Top: Original and Reconstructed) ---
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



                # --- Loss Plots (Bottom 2x2 Grid) ---
                # Titles and data for each loss plot
                loss_data = [gaussian_recon_losses, recon_losses, lpips_losses, disc_losses, adversarial_losses]
                loss_titles = ['Gaussian Loss', 'L1 Loss', 'LPIPS Loss', 'Discriminator Losses', 'Discriminator Losses']
                loss_colors = ['orange', 'blue', 'green', 'red', 'orange']
                labels = [None, None, None, 'Discriminator Loss', 'Adversarial Loss']

                for i in range(len(loss_data)):
                    loss_axes[min(i,3)].plot(loss_data[i], color=loss_colors[i], label=labels[i])
                    loss_axes[min(i,3)].set_title(loss_titles[i])
                    loss_axes[min(i,3)].set_xlabel('Steps')
                    # loss_axes[min(i,3)].set_xlim(left=95)
                    loss_axes[min(i,3)].set_ylabel('Loss')
                    loss_axes[min(i,3)].grid(True, linestyle='--', alpha=0.7)
                    loss_axes[min(i,3)].set_xscale('log')  
                    if i!=0:
                        loss_axes[min(i,3)].set_yscale('log')  

                    
                    if i==(len(loss_data)-1):
                        loss_axes[min(i,3)].legend()

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