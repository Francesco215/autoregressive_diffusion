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
    # Initialize the LTX-Video VAE with the provided config
    vae = AutoencoderKLLTXVideo(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=[4, 3, 3, 3, 4],
        patch_size=4,
        patch_size_t=1,
        encoder_causal=True,
        decoder_causal=True,
        resnet_norm_eps=1e-06,
        scaling_factor=1.0,
        spatio_temporal_scaling=[True, True, True, False]
    )
    
    # Optionally load pre-trained weights if available
    # vae.load_state_dict(torch.load("path/to/downloaded/weights.pt"))
    
    return vae.to(device)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae = load_ltx_video_vae(device)

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

    # Check if we can explicitly access the first frame latent
    # This is just a test to see the structure - it might fail
    try:
        print("Attempting to examine latent structure...")
        if hasattr(latent_dist, 'first_frame'):
            print(f"First frame latent shape: {latent_dist.first_frame.shape}")
        else:
            print("No explicit first_frame attribute in latent_dist")
            
        # Try to inspect if there's any pattern in the latent representation
        print(f"First frame of latent: {latent[:,:,0,:,:].shape}")
        print(f"Rest of frames of latent: {latent[:,:,1:,:,:].shape}")
    except Exception as e:
        print(f"Error while inspecting latent: {str(e)}")

def video_dwt_loss(x, y):
    """
    Compute DWT loss between original and reconstructed videos
    x, y: tensors of shape [B, C, T, H, W]
    """
    # Initialize 2D DWT transform and move to the same device as input tensors
    dwt = DWTForward(J=1, mode='zero', wave='db1').to(x.device)
    
    batch_size, channels, time_steps, height, width = x.shape
    dwt_loss = 0.0
    
    # For each time step, compute 2D DWT
    for t in range(time_steps):
        # Get current frame
        x_t = x[:, :, t]  # [B, C, H, W]
        y_t = y[:, :, t]  # [B, C, H, W]
        
        # Apply DWT transform
        x_coeffs = dwt(x_t)  # Returns low-pass and list of high-pass coefficients
        y_coeffs = dwt(y_t)
        
        # Calculate L1 loss on low-frequency components
        dwt_loss += F.l1_loss(x_coeffs[0], y_coeffs[0])
        
        # Calculate L1 loss on high-frequency components (horizontal, vertical, diagonal)
        for i in range(len(x_coeffs[1])):
            dwt_loss += F.l1_loss(x_coeffs[1][i], y_coeffs[1][i])
    
    return dwt_loss / time_steps

def perceptual_loss(x, y):
    """
    Compute LPIPS perceptual loss between original and reconstructed videos
    x, y: tensors of shape [B, C, T, H, W]
    """
    batch_size, channels, time_steps, height, width = x.shape
    total_loss = 0.0
    
    # For each time step, compute LPIPS
    for t in range(time_steps):
        # Get current frame and ensure it's in the right format (normalized to [-1,1])
        x_t = x[:, :, t]  # [B, C, H, W]
        y_t = y[:, :, t]  # [B, C, H, W]
        
        # Calculate perceptual loss - make sure lpips_loss_fn is on the same device
        # as the input tensors
        p_loss = lpips_loss_fn(x_t, y_t)
        total_loss += p_loss.mean()
    
    return total_loss / time_steps

def reconstruction_gan_losses(discriminator, real_frames, reconstructed_frames):

    batch_size = real_frames.shape[0]
    device = real_frames.device

    real_recon_pairs = torch.cat([real_frames, reconstructed_frames], dim=2)  # Concat along time dimension
    recon_real_pairs = torch.cat([reconstructed_frames, real_frames], dim=2)
  
    real_recon_logits = discriminator(real_recon_pairs)  # Should predict 1 (first is real)
    recon_real_logits = discriminator(recon_real_pairs)  # Should predict 0 (second is real)

    ones = torch.ones(batch_size, *real_recon_logits.shape[2:], device=device, dtype=torch.long)
    zeros = torch.zeros(batch_size, *recon_real_logits.shape[2:], device=device, dtype=torch.long)

    gen_loss_real_recon = F.cross_entropy(real_recon_logits, zeros)/np.log(2)  # Want discriminator to think recon is real
    gen_loss_recon_real = F.cross_entropy(recon_real_logits, ones)/np.log(2)   # Want discriminator to think real is recon
    adversarial_loss = (gen_loss_real_recon + gen_loss_recon_real) / 2

    disc_loss_real_recon = F.cross_entropy(real_recon_logits, ones)/np.log(2)   # Should predict real is real
    disc_loss_recon_real = F.cross_entropy(recon_real_logits, zeros)/np.log(2)  # Should predict recon is recon
    discriminator_loss = (disc_loss_real_recon + disc_loss_recon_real) / 2
    
    return adversarial_loss, discriminator_loss


#%%
torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"

    batch_size = 16
    micro_batch_size = 2
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
    discriminator = MixedDiscriminator(in_channels = 3, block_out_channels=(32,)).to(device)
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
    
    # Initialize LPIPS loss function


    
    
    # Training loop
    for _ in range(10):
        pbar = tqdm(enumerate(dataloader), total=total_number_of_steps)
        for batch_idx, micro_batch in pbar:
            with torch.no_grad():
                frames, _ = micro_batch  # Ignore actions and reward for this VggAE training
                frames = frames.float() / 127.5 - 1  # Normalize to [-1, 1]
                frames = einops.rearrange(frames, 'b t h w c-> b c t h w').to(device)

    
            latent = vae.encode(frames).latent_dist.sample()
      
            decoder_output = vae.decode(latent)
            recon = decoder_output.sample
            posterior = vae.encode(frames).latent_dist
            mean = posterior.mean
            logvar = posterior.logvar
            
           
            uniform_logvar = logvar.mean(dim=1, keepdim=True)
            uniform_logvar = uniform_logvar.expand_as(logvar)
            kl_loss = -0.5 * torch.mean(1 + uniform_logvar - mean.pow(2) - uniform_logvar.exp())
            
            logits = discriminator(recon)
            targets = torch.ones(logits.shape[0], *logits.shape[2:], device=device, dtype=torch.long)
            

            adversarial_loss, loss_disc = reconstruction_gan_losses(discriminator, frames, recon)
            
   
            # Pixel reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, frames)
            
            # DWT loss
            dwt_loss = video_dwt_loss(recon, frames)
            
            # Perceptual loss (LPIPS)
            perceptual_loss_value = perceptual_loss(recon, frames)
            
            # Total VAE loss
            vae_loss = recon_loss + 0.1 * dwt_loss + 0.1 * perceptual_loss_value + kl_loss*1e-5 + adversarial_loss*1e-5
            vae_loss.backward()

            # Update VAE parameters
            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()
            
            # Train discriminator
            loss_disc.backward()
            
            # Update discriminator parameters
            if batch_idx % (batch_size//micro_batch_size) == 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
                optimizer_disc.step()
                scheduler_disc.step()
                optimizer_disc.zero_grad()
            
            # Update progress bar
            pbar.set_postfix_str(f"pixel: {recon_loss.item():.4f}, dwt: {dwt_loss.item():.4f}, lpips: {perceptual_loss_value.item():.4f}, kl: {kl_loss.item():.4f}, disc: {loss_disc.item():.4f}, adv: {adversarial_loss.mean().item():.4f}")
            
            # Store losses
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            adversarial_losses.append(adversarial_loss.mean().item())
            disc_losses.append(loss_disc.item())

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