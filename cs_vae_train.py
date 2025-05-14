#%%
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW
from diffusers import AutoencoderKLLTXVideo
import pywt
import pytorch_wavelets
from pytorch_wavelets import DWTForward, DWTInverse
import lpips

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from edm2.cs_dataloading import CsCollate, CsDataset
from edm2.vae import VAE, MixedDiscriminator

import os
import tempfile

# Set custom temporary directory
os.environ['TMPDIR'] = '/mnt/mnemo9/mpelus/'
# Optional: Also set for tempfile module
tempfile.tempdir = '/mnt/mnemo9/mpelus/'
#%%
# Load the LTX-Video VAE
def load_ltx_video_vae(device):
    # Configuration for a 3-block encoder/decoder architecture
    num_blocks = 3
    # Encoder settings
    enc_block_out_channels = [128, 256, 512]        # Length: num_blocks
    enc_layers_per_block = [4, 3, 3, 4]             # Length: num_blocks + 1
    enc_spatio_temporal_scaling = [True, True, True]# Length: num_blocks
    # Decoder settings (to match encoder structure)
    # LTXVideoDecoder3d internally reverses them.
    dec_block_out_channels = list(enc_block_out_channels)           # e.g., [128, 256, 512]
    # For symmetric layer counts, decoder_layers_per_block can mirror encoder_layers_per_block.
    # LTXVideoDecoder3d expects this in [L_outer, ..., L_inner, L_mid] format, then reverses it.
    dec_layers_per_block = list(enc_layers_per_block)               # e.g., [4, 3, 3, 4]
    dec_spatio_temporal_scaling = list(enc_spatio_temporal_scaling) # e.g., [True, True, True]
    # Tuple lengths dependent on num_blocks for the decoder:
    # inject_noise: num_blocks + 1
    # others: num_blocks
    dec_inject_noise = (True,) * (num_blocks + 1)             # (True, True, True, True)
    dec_upsample_residual = (False,) * num_blocks             # (False, False, False)
    dec_upsample_factor = (1,) * num_blocks                   # (1, 1, 1)

    vae = AutoencoderKLLTXVideo(
        in_channels=3,
        out_channels=3,
        latent_channels=128,

        # Encoder parameters
        block_out_channels=enc_block_out_channels,
        layers_per_block=enc_layers_per_block,
        spatio_temporal_scaling=enc_spatio_temporal_scaling,

        # Decoder parameters
        decoder_block_out_channels=dec_block_out_channels,
        decoder_layers_per_block=dec_layers_per_block,
        decoder_spatio_temporal_scaling=dec_spatio_temporal_scaling,
        decoder_inject_noise=dec_inject_noise,
        upsample_residual=dec_upsample_residual,
        upsample_factor=dec_upsample_factor,

        patch_size=4,
        patch_size_t=1,
        encoder_causal=True,
        decoder_causal=True,  # This is passed to LTXVideoDecoder3d as is_causal
        resnet_norm_eps=1e-06,
        scaling_factor=1.0,
        # timestep_conditioning=True, # Set if needed
    )
    return vae.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = load_ltx_video_vae(device)

def test():
    # Create a dummy video tensor with multiple frames
    # [batch, channels, time, height, width]
    dummy_frames = torch.randn(2, 3, 17, 64, 64).to(device)
    print(f"Input shape: {dummy_frames.shape}")

    # Pass through encoder
    encoded = vae.encode(dummy_frames)
    latent_dist = encoded.latent_dist
    latent = latent_dist.sample()
    print(f"Latent shape: {latent.shape}")

    # Pass through decoder
    decoded = vae.decode(latent)
    recon = decoded.sample
    print(f"Output shape: {recon.shape}")

def video_dwt_loss(x, y):
    """
    Compute DWT loss between original and reconstructed videos
    x, y: tensors of shape [B, C, T, H, W]
    """
    # Initialize 2D DWT transform and move to the same device as input tensors
    dwt = DWTForward(J=1, mode='zero', wave='db1').to(x.device)
    
    batch_size, channels, time_steps, height, width = x.shape
    
    # Reshape to [B*T, C, H, W] for batch processing
    x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    y_reshaped = y.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    
    # Apply DWT transform to all frames at once
    x_coeffs = dwt(x_reshaped)  # Returns low-pass and list of high-pass coefficients
    y_coeffs = dwt(y_reshaped)
    
    # Calculate L1 loss on low-frequency components
    dwt_loss = F.l1_loss(x_coeffs[0], y_coeffs[0])
    
    # Calculate L1 loss on high-frequency components (horizontal, vertical, diagonal)
    for i in range(len(x_coeffs[1])):
        dwt_loss += F.l1_loss(x_coeffs[1][i], y_coeffs[1][i])
    
    return dwt_loss

def perceptual_loss(x, y):
    """
    Compute LPIPS perceptual loss between original and reconstructed videos
    x, y: tensors of shape [B, C, T, H, W]
    """
    batch_size, channels, time_steps, height, width = x.shape
    x = torch.clamp(x, -1.0, 1.0)
    y = torch.clamp(y, -1.0, 1.0)
    # Reshape to combine batch and time dimensions
    x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)  # [B*T, C, H, W]
    y_reshaped = y.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)  # [B*T, C, H, W]
    
    # Calculate perceptual loss for all frames at once
    p_loss = lpips_loss_fn(x_reshaped, y_reshaped)
    
    # Reshape loss back to [B, T] and take mean
    p_loss = p_loss.reshape(batch_size, time_steps).mean()
    
    return p_loss


#%%
torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"

    batch_size = 16
    micro_batch_size = 1
    clip_length = 17 
    
    assert (clip_length - 1) % 8 == 0, 'clip length -1 must be divisible by 8.'
    
    # Hyperparameters
    # latent_channels = 8
    # n_res_blocks = 2
    # channels = [3, 32, 32, latent_channels]

    # Initialize models
    vae = load_ltx_video_vae(device)
    #vae = VAE(channels = channels, n_res_blocks=n_res_blocks, spatial_compressions=[1,4,4]).to(device)
    # vae = VAE.load_from_pretrained('saved_models/vae_cs_4264.pt').to(device)
    # Example instantiation
    discriminator = MixedDiscriminator(in_channels = 6, block_out_channels=(32,)).to(device)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    
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
    base_lr = 1e-4 #4
    optimizer_vae = AdamW(vae.parameters(), lr=base_lr, eps=1e-8)
    optimizer_disc = AdamW(discriminator.parameters(), lr=base_lr, eps=1e-8)
    optimizer_vae.zero_grad()
    optimizer_disc.zero_grad()

    # Add exponential decay schedule
    gamma = 0.1 ** (1 / total_number_of_steps)  # Decay factor so lr becomes 0.1 * initial_lr after 40,000 steps
    scheduler_vae = lr_scheduler.ExponentialLR(optimizer_vae, gamma=gamma)
    scheduler_disc = lr_scheduler.ExponentialLR(optimizer_disc, gamma=gamma)
    losses = []


    recon_losses,  kl_losses, disc_losses, adversarial_losses= [], [], [], []

    #%%
# Assuming the same setup as before

    # Training loop
    for epoch in range(10):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        grad_accum_steps = batch_size // micro_batch_size

        for batch_idx, micro_batch in pbar:
            frames_raw, _ = micro_batch
            frames = frames_raw.float() / 127.5 - 1
            frames = einops.rearrange(frames, 'b t h w c -> b c t h w').to(device)
            current_batch_size = frames.shape[0]

            # Step 0: Common forward pass
            posterior = vae.encode(frames).latent_dist 
            latent_sampled = posterior.sample()
            recon_g = vae.decode(latent_sampled).sample

            # Step 1: VAE/Generator losses
            # KL loss
            mean = posterior.mean
            logvar = posterior.logvar
            uniform_logvar = logvar.mean(dim=1, keepdim=True).expand_as(logvar)
            kl_loss = -0.5 * torch.mean(1 + uniform_logvar - mean.pow(2) - uniform_logvar.exp())
            
            # Reconstruction losses
            recon_loss_for_vae = F.mse_loss(recon_g, frames)
            dwt_loss = video_dwt_loss(recon_g, frames)
            perceptual_loss_value = perceptual_loss(recon_g, frames)
            
            # Adversarial loss for VAE/generator
            frames_recon_g_pairs_adv = torch.cat([frames, recon_g], dim=1)
            recon_g_frames_pairs_adv = torch.cat([recon_g, frames], dim=1)
            
            frames_recon_g_logits_adv = discriminator(frames_recon_g_pairs_adv)
            recon_g_frames_logits_adv = discriminator(recon_g_frames_pairs_adv)

            spatial_dims_disc_out = frames_recon_g_logits_adv.shape[2:]
            ones_targets = torch.ones(current_batch_size, *spatial_dims_disc_out, device=device, dtype=torch.long)
            zeros_targets = torch.zeros(current_batch_size, *spatial_dims_disc_out, device=device, dtype=torch.long)
            
            # G wants D to misclassify
            gen_loss_frames_recon = F.cross_entropy(frames_recon_g_logits_adv, zeros_targets) / np.log(2)
            gen_loss_recon_frames = F.cross_entropy(recon_g_frames_logits_adv, ones_targets) / np.log(2)
            adversarial_loss_for_vae = (gen_loss_frames_recon + gen_loss_recon_frames) / 2
            
            # Total VAE loss
            total_vae_loss = recon_loss_for_vae 

            # Step 2: Discriminator losses
            recon_g_detached = recon_g.detach() 
            
            real_frames_recon_d_pairs = torch.cat([frames, recon_g_detached], dim=1)
            recon_d_real_frames_pairs = torch.cat([recon_g_detached, frames], dim=1)
            
            real_frames_recon_d_logits = discriminator(real_frames_recon_d_pairs)
            recon_d_real_frames_logits = discriminator(recon_d_real_frames_pairs)
            
            # D wants correct classification
            disc_loss_type1 = F.cross_entropy(real_frames_recon_d_logits, ones_targets) / np.log(2)
            disc_loss_type0 = F.cross_entropy(recon_d_real_frames_logits, zeros_targets) / np.log(2)
            total_loss_disc = (disc_loss_type1 + disc_loss_type0) / 2
            
            # Step 3: Updates
            # Update VAE/Generator
            optimizer_vae.zero_grad()
            total_vae_loss.backward()
            if batch_idx % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer_vae.step()
                scheduler_vae.step()

            # Update Discriminator
            optimizer_disc.zero_grad()
            total_loss_disc.backward()
            if batch_idx % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_disc.step()
                scheduler_disc.step()
                
            # Update progress bar & store losses
            pbar.set_postfix_str(
                f"pixel: {recon_loss_for_vae.item():.4f}, kl: {kl_loss.item():.4f}, "
                f"disc: {total_loss_disc.item():.4f}, adv: {adversarial_loss_for_vae.item():.4f}"
            )
       
            
            # Store losses
            recon_losses.append(recon_loss_for_vae.item())
            kl_losses.append(kl_loss.item())
            adversarial_losses.append(adversarial_loss_for_vae.item())
            disc_losses.append(total_loss_disc.item())

            # if batch_idx == 500:
            #     adv_multiplier = 5e-2

            # Visualization every 100 steps
            if batch_idx % 100 == 0 and batch_idx > 0:
                # Create a figure with a custom layout: 3 sections (2 rows for frames, 3 loss plots)
                fig = plt.figure(figsize=(15, 12))

                # Top section: 2 rows for original and reconstructed frames (2x5 grid)
                gs_top = plt.GridSpec(2, 5, figure=fig, top=0.95, bottom=0.55, left=0.1, right=0.9)
                orig_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]  # Row 0: Original frames
                recon_axes = [fig.add_subplot(gs_top[1, i]) for i in range(5)]  # Row 1: Reconstructed frames

                # Bottom section: 3 subplots for losses (1x3 grid)
                gs_bottom = plt.GridSpec(1, 3, figure=fig, top=0.45, bottom=0.05, left=0.1, right=0.9, wspace=0.3)
                loss_axes = [
                    fig.add_subplot(gs_bottom[0, 0]),  # Left: Recon Loss
                    fig.add_subplot(gs_bottom[0, 1]),  # Middle: KL Loss
                    fig.add_subplot(gs_bottom[0, 2])   # Right: Combined Disc and Adversarial Loss
                ]

                # --- Video Frames (Top: Original and Reconstructed) ---
                with torch.no_grad():
                    # Detach and denormalize frames and reconstructions
                    frames_denorm = (frames.cpu() + 1) / 2  # Shape: (batch, channels, time, height, width)
                    recon_denorm = (recon_g.cpu() + 1) / 2    # Shape: (batch, channels, time, height, width)

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
                

                # --- Loss Plots (Bottom Row) ---
                # Plot reconstruction loss
                loss_axes[0].plot(recon_losses, color='blue')
                loss_axes[0].set_title('Reconstruction Loss')
                loss_axes[0].set_yscale('log')  
                loss_axes[0].set_xscale('log')  
                loss_axes[0].set_xlabel('Steps')
                loss_axes[0].set_xlim(left=10)
                loss_axes[0].set_ylabel('Loss')
                loss_axes[0].grid(True, linestyle='--', alpha=0.7)
                
                # Plot KL loss
                loss_axes[1].plot(kl_losses, color='green')
                loss_axes[1].set_title('KL Loss')
                loss_axes[1].set_yscale('log')  
                loss_axes[1].set_xscale('log')  
                loss_axes[1].set_xlabel('Steps')
                loss_axes[1].set_xlim(left=10)
                loss_axes[1].set_ylabel('Loss')
                loss_axes[1].grid(True, linestyle='--', alpha=0.7)
                
                # Plot discriminator and adversarial losses together
                loss_axes[2].plot(disc_losses, color='red', label='Discriminator Loss')
                loss_axes[2].plot(adversarial_losses, color='purple', label='Adversarial Loss')
                loss_axes[2].set_title('Discriminator & Adversarial Losses')
                loss_axes[2].set_yscale('log')  
                loss_axes[2].set_xscale('log')  
                loss_axes[2].set_xlabel('Steps')
                loss_axes[2].set_xlim(left=10)
                loss_axes[2].set_ylabel('Loss')
                loss_axes[2].grid(True, linestyle='--', alpha=0.7)
                loss_axes[2].legend()

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