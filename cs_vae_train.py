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

# Import LPIPS (commented out - replaced with segmentation loss)
# import lpips 

# Import FastSAM for segmentation loss
from ultralytics import FastSAM

from edm2.cs_dataloading import CsCollate, CsDataset
from edm2.vae import VAE, MixedDiscriminator
# from edm2.utils import GaussianLoss  # Replaced with segment-aware version

os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/mnt/mnemo9/mpelus/experiments/autoregressive_diffusion/.torchinductor_cache'

torch._dynamo.config.recompile_limit = 64
# torch.autograd.set_detect_anomaly(True)

def segLoss(r_mean, r_logvar, target_frames, fastsam_model):
    """
    Segment-aware Gaussian loss function
    r_mean, r_logvar, target_frames: [B, C, T, H, W] in range [-1, 1]
    Returns: (loss, segments_list)
    """
    # Flatten for FastSAM
    target_flat = einops.rearrange(target_frames, 'b c t h w -> (b t) c h w')
    r_mean_flat = einops.rearrange(r_mean, 'b c t h w -> (b t) c h w')
    r_logvar_flat = einops.rearrange(r_logvar, 'b c t h w -> (b t) c h w')
    
    with torch.no_grad():
        # Convert target from [-1, 1] to [0, 1] for FastSAM
        target_normalized = (target_flat + 1) / 2
        results = fastsam_model(target_normalized, device="cuda", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
    
    # Compute losses
    frame_losses = []
    segments_list = []
    
    for i, result in enumerate(results):
        masks = result.masks.data.clone().detach().to(r_mean.device).float()
        
        # Create leftover mask for pixels not covered by any segment
        if len(masks) > 0:
            combined_mask = masks.sum(dim=0)  # Sum all masks
            leftover_mask = (combined_mask == 0).float()  # Pixels not covered by any mask
            
            # Add leftover mask if it has any pixels
            if leftover_mask.sum() > 0:
                masks = torch.cat([masks, leftover_mask.unsqueeze(0)], dim=0)
        else:
            # If no segments found, treat entire frame as one segment
            h, w = target_flat[i].shape[-2:]
            leftover_mask = torch.ones(h, w, device=r_mean.device, dtype=torch.float)
            masks = leftover_mask.unsqueeze(0)
        
        segments_list.append(masks.cpu())
        
        frame_mean = r_mean_flat[i]
        frame_logvar = r_logvar_flat[i]
        frame_target = target_flat[i]
        
        segment_losses = []
        for mask in masks:
            mask_expanded = mask.unsqueeze(0)
            
            masked_mean = frame_mean * mask_expanded
            masked_logvar = frame_logvar * mask_expanded  
            masked_target = frame_target * mask_expanded
            
            # Compute the main loss terms per pixel
            loss_per_pixel = ((masked_logvar + (masked_mean - masked_target)**2 * torch.exp(-masked_logvar)) * 0.5)
            
            # Average over the segment, then add the constant once
            segment_loss = loss_per_pixel.sum() / (mask.sum() * frame_mean.shape[0] + 1e-8) + 0.918
            segment_losses.append(segment_loss)
        
        frame_losses.append(torch.stack(segment_losses).mean())
    
    return torch.stack(frame_losses).mean(), segments_list

if __name__=="__main__":
   device = "cuda"

   batch_size = 1
   micro_batch_size = 1
   clip_length = 32

   # Hyperparameters
   latent_channels = 8
   n_res_blocks = 4 #3
   channels = [3, 32, 128, 512, latent_channels] #16, 64, 256

   # Initialize models
   vae = VAE(channels = channels, n_res_blocks=n_res_blocks, spatial_compressions=[1,2,2,2], time_compressions=[1,2,2,1]).to(device)
   #vae = vae.to(torch.float16)
   #vae = VAE.from_pretrained('saved_models/vae_cs_15990.pt').to(device)
   #discriminator = MixedDiscriminator().to(device)
   #vae, discriminator = torch.compile(vae), torch.compile(discriminator)
   vae = torch.compile(vae)
   
   # Initialize FastSAM for segmentation loss
   fastsam = FastSAM("FastSAM-s.pt")
   
   dataset = CsDataset(clip_size=clip_length, remote='s3://counter-strike-data/original/', local = '/mnt/mnemo9/mpelus/experiments/autoregressive_diffusion/streaming_dataset/cs_vae',batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
   
   dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsCollate(clip_length), num_workers=8, shuffle=False)
   total_number_of_steps = len(dataloader)//micro_batch_size

   vae_params = sum(p.numel() for p in vae.parameters())
   #discriminator_params = sum(p.numel() for p in discriminator.parameters())
   print(f"Number of vae parameters: {vae_params//1e3}K")
   #print(f"Number of discriminator parameters: {discriminator_params//1e3}K")

   # Define optimizers
   base_lr = 1e-4
   optimizer_vae = AdamW((p for p in vae.parameters() if p.requires_grad), lr=base_lr, eps=1e-8)
   #optimizer_disc = AdamW(discriminator.parameters(), lr=base_lr, eps=1e-8)
   optimizer_vae.zero_grad()
   #optimizer_disc.zero_grad()

   # --- Scheduler Definition ---
   warmup_steps = 100
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
   #scheduler_disc = lr_scheduler.LambdaLR(optimizer_disc, lr_lambda)

   # Initialize LPIPS loss function (commented out - replaced with segmentation loss)
   # lpips_loss_fn = lpips.LPIPS(net='alex')#.to(torch.float16)
   # if torch.cuda.is_available():
   #     lpips_loss_fn.cuda()

   # Store losses
   seg_losses, l1_recon_losses = [], [] # adversarial_losses, discriminator_losses = [], [], [], [], []
   # gaussian_recon_losses, lpips_losses removed - replaced with seg_losses

   #%%
   # Training loop
   for _ in range(10):
       pbar = tqdm(enumerate(dataloader), total=total_number_of_steps)
       for batch_idx, micro_batch in pbar:
           with torch.no_grad():
               frames, _ = micro_batch # Ignore actions and reward for this VggAE training
               frames = frames.float() / 127.5 - 1 # Normalize to [-1, 1]
               
               frames = einops.rearrange(frames, 'b t h w c-> b c t h w').to(device)
               #frames = frames.to(torch.float16)

           # VAE forward pass
           # r_mean (reconstruction mean): This is your hat_x
           # r_logvar (reconstruction log variance): Used for GaussianLoss
           # mean (latent mean), logvar (latent log variance): For KL divergence
           r_mean, r_logvar, mean, _ = vae(frames)

           # VAE losses
           # gaussian_loss = GaussianLoss(r_mean, r_logvar, frames)  # Replaced with segment-aware version
           seg_loss, segments = segLoss(r_mean, r_logvar, frames, fastsam)
           l1_loss = F.l1_loss(r_mean, frames)

           # --- LPIPS Calculation (commented out - replaced with segmentation loss) ---
           # # Reshape frames and r_mean from [B, C, T, H, W] to [B*T, C, H, W]
           # # so that LPIPS can process them as individual images.
           # # Your frames are C=3, and T=32, so we need to flatten T into the batch dimension.
          
           # frames_flat = torch.clip(einops.rearrange(frames, 'b c t h w -> (b t) c h w'), -1, 1)
           # r_mean_flat = torch.clip(einops.rearrange(r_mean, 'b c t h w -> (b t) c h w'), -1, 1)

           # # Calculate LPIPS loss for each frame and then take the mean
           # raw_lpips_per_frame = lpips_loss_fn(r_mean_flat, frames_flat)  # Shape: [B*T]
           # eps = 1e-8
           # log_lpips_per_frame = torch.log(raw_lpips_per_frame + eps)
           # lpips_loss = log_lpips_per_frame.mean()
           #adversarial_loss = discriminator.vae_loss(frames,r_mean)

           main_loss = seg_loss # + adversarial_loss*0.01 #*0.05*min(1,batch_idx/total_number_of_steps)
           # main_loss = gaussian_loss + lpips_loss*0.1  # Old loss computation
           main_loss.backward()

           if batch_idx % (batch_size//micro_batch_size) == 0 and batch_idx!=0:
               nn.utils.clip_grad_norm_(vae.parameters(), 0.5)
               optimizer_vae.step()
               scheduler_vae.step()
               optimizer_vae.zero_grad()
               #optimizer_disc.zero_grad()

           #loss_disc = discriminator.discriminator_loss(frames, r_mean)
           #loss_disc.backward()

           # Update discriminator
           # if batch_idx % (batch_size//micro_batch_size) == 0:
           #     nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
           #     optimizer_disc.step()
           #     scheduler_disc.step()
           #     optimizer_disc.zero_grad()

           pbar.set_postfix_str(f"seg_loss: {seg_loss.item():.4f}, l1_recon: {l1_loss.item():.4f}, current_lr: {optimizer_vae.param_groups[0]['lr']:.6f}")#, disc_loss: {loss_disc.item():.4f}
           seg_losses.append(seg_loss.item()) # Store all loss components for plotting
           l1_recon_losses.append(l1_loss.item())
           # gaussian_recon_losses.append(gaussian_loss.item()) # Replaced with seg_losses
           # lpips_losses.append(lpips_loss.item()) # Replaced with seg_losses
           # adversarial_losses.append(adversarial_loss.item()) 
           # discriminator_losses.append(loss_disc.item()) 

           if batch_idx % 100 == 0 and batch_idx > 0:
               fig = plt.figure(figsize=(15, 22)) # Increased figure height for the new segmentation row
               fig.suptitle(f"VAE Training Progress - VAE Parameters: {vae_params//1e6}M", fontsize=16)
               # Top section: 4 rows for original, reconstructed (mean), uncertainty heatmaps, and segmentation masks
               gs_top = plt.GridSpec(4, 5, figure=fig, top=0.95, bottom=0.5, left=0.1, right=0.9) # Added one more row
               orig_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
               recon_mean_axes = [fig.add_subplot(gs_top[1, i]) for i in range(5)]
               uncertainty_axes = [fig.add_subplot(gs_top[2, i]) for i in range(5)]
               segment_axes = [fig.add_subplot(gs_top[3, i]) for i in range(5)] # New row for segmentation masks

               # Bottom section: 1x2 for loss plots (removed LPIPS plot)
               gs_bottom = plt.GridSpec(1, 2, figure=fig, top=0.45, bottom=0.1, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
               loss_axes = [
                   fig.add_subplot(gs_bottom[0, 0]),
                   fig.add_subplot(gs_bottom[0, 1]),
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

                       # Segmentation Masks
                       segment_axes[i].imshow(frames_denorm[idx]) # Base image
                       if idx < len(segments) and segments[idx] is not None:
                           # Create a composite mask visualization
                           segment_mask = segments[idx] # [N_segments, H, W]
                           if segment_mask.shape[0] > 0:
                               # Create colored overlay for different segments
                               colored_mask = torch.zeros(segment_mask.shape[1], segment_mask.shape[2], 3)
                               colors = plt.cm.Set3(np.linspace(0, 1, min(segment_mask.shape[0], 12)))[:, :3]  # Different colors
                               for seg_idx, mask in enumerate(segment_mask):
                                   color = colors[seg_idx % len(colors)]
                                   for c in range(3):
                                       colored_mask[:, :, c] += mask * color[c]
                               
                               segment_axes[i].imshow(colored_mask, alpha=0.5)
                       segment_axes[i].set_title(f"Segments t={idx}")
                       segment_axes[i].axis('off')

               # Plot Segmentation loss (replaces Gaussian loss)
               loss_axes[0].plot(seg_losses, label="Segmentation Loss\n(segment-aware Gaussian)", color="red")
               loss_axes[0].set_title("Segmentation Losses")
               loss_axes[0].set_xscale("log")
               loss_axes[0].set_xlabel("Steps")
               loss_axes[0].set_ylabel("Loss")
               if len(seg_losses) > 95:
                   loss_axes[0].set_ybound(upper = seg_losses[95])
                   loss_axes[0].set_xbound(lower = 95)
               loss_axes[0].grid(True)

               loss_axes[1].plot(l1_recon_losses, label="L1 Recon Loss\n(we don't optimize for this)", color="blue")
               loss_axes[1].set_title("L1 Reconstruction Losses")
               loss_axes[1].set_yscale("log")
               loss_axes[1].set_xscale("log")
               loss_axes[1].set_xlabel("Steps")
               loss_axes[1].set_ylabel("Loss")
               if len(l1_recon_losses) > 95:
                   loss_axes[1].set_ybound(upper = l1_recon_losses[95])
                   loss_axes[1].set_xbound(lower = 95)
               loss_axes[1].grid(True)

               # LPIPS plot removed - replaced with segmentation loss
               # loss_axes[2].plot(lpips_losses, label="LPIPS Loss\n(we optimize for this)", color="green")
               # loss_axes[2].set_title("LPIPS Loss")
               # loss_axes[2].set_xscale("log")
               # loss_axes[2].set_xlabel("Steps")
               # loss_axes[2].set_ylabel("Loss")
               # loss_axes[2].grid(True)

               plt.tight_layout()
               os.makedirs("images_training", exist_ok=True)
               plt.savefig(f"images_training/{batch_idx}.png")
               plt.close()

           if batch_idx % (total_number_of_steps // 10) == 0 and batch_idx != 0:
               os.makedirs("saved_models", exist_ok=True)
               vae.save_to_state_dict(f'saved_models/vae_cs_{batch_idx}.pt')

   print("Finished Training")
   #%%