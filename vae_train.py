#%%
import einops
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



from edm2.gym_dataloader import GymDataGenerator, gym_collate_function
from edm2.vae import VAE, EncoderDecoder
from edm2.mars import MARS
torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"
    model_id="stabilityai/stable-diffusion-2-1"

    batch_size = 2
    state_size = 48 
    total_number_of_steps = 4_000
    training_steps = total_number_of_steps * batch_size

    
    # Hyperparameters
    latent_channels = 16
    n_res_blocks = 2

    # Initialize models
    vae = VAE(latent_channels=latent_channels, n_res_blocks=n_res_blocks).to(device)
    discriminator = EncoderDecoder(latent_channels = 2, n_res_blocks=n_res_blocks, time_compressions=[1, 2, 4], spatial_compressions=[1, 4, 4], type='discriminator').to(device)
    
    dataset = GymDataGenerator(state_size, original_env, training_steps, autoencoder_time_compression = 4)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=gym_collate_function, num_workers=16)

    vae_params = sum(p.numel() for p in vae.parameters())
    discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of vae parameters: {vae_params//1e3}K")
    print(f"Number of discriminator parameters: {discriminator_params//1e3}K")
    # sigma_data = 0.434
    sigma_data = 1.

    # Define optimizers
    base_lr = 1e-3
    optimizer_vae = MARS(vae.parameters(), lr=base_lr, eps=1e-4)
    optimizer_disc = MARS(discriminator.parameters(), lr=base_lr*2e-2, eps=1e-4)

    # Add exponential decay schedule
    gamma = 0.01 ** (1 / total_number_of_steps)  # Decay factor so lr becomes 0.1 * initial_lr after 40,000 steps
    scheduler_vae = lr_scheduler.ExponentialLR(optimizer_vae, gamma=gamma)
    scheduler_disc = lr_scheduler.ExponentialLR(optimizer_disc, gamma=gamma)
    losses = []

    resume_training_run = None
    pbar = tqdm(enumerate(dataloader), total=total_number_of_steps)

    #%%
    # Training loop
    for batch_idx, batch in pbar:
        with torch.no_grad():
            frames, _, _ = batch  # Ignore actions and reward for this VAE training
            frames = frames.float() / 127.5 - 1  # Normalize to [-1, 1]
            frames = einops.rearrange(frames, 'b t h w c-> b c t h w').to(device)

        # VAE forward pass
        recon, mean, logvar, _ = vae(frames)

        # in theory the mean should be only with respect to the batch size
        group_mean = mean.mean(dim=(0,2))
        group_var = mean.var(dim=(0,2))

        kl_group = - 0.5 * (1 + group_var.log() - group_mean.pow(2) - group_var).sum(dim=0).mean()

        # VAE losses
        recon_loss = F.mse_loss(recon, frames, reduction='mean')

        # kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1).mean()
        kl_loss = -0.5 * (1 + logvar - logvar.exp()).sum(dim=1).mean()

        # Compute all discriminator outputs
        frames = torch.cat([frames, recon], dim=0)
        logits, _ = discriminator(frames)
        logits = einops.rearrange(logits, 'b c t h w -> (b t h w) c')
        targets = torch.cat([torch.ones(logits.shape[0]//2, device=device, dtype=torch.long), torch.zeros(logits.shape[0]//2, device=device, dtype=torch.long)], dim=0)
        loss_disc = F.cross_entropy(logits, targets)/np.log(2)

        vae_loss = recon_loss + kl_group*1e-3 + kl_loss*1e-3 - loss_disc*1e-2

        # Update VAE
        optimizer_vae.zero_grad()
        vae_loss.backward(retain_graph=True) #am i wasting resources?
        optimizer_vae.step()
        scheduler_vae.step()  # Step the VAE scheduler

        # Update discriminator
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
        scheduler_disc.step()
        
        pbar.set_postfix_str(f"MSE loss: {recon_loss.item():.4f}, KL group loss: {kl_group.item():.4f} KL loss: {kl_loss.item():.4f}, Discr Loss: {loss_disc.item():.4f}")
        # Visualization every 100 steps
        if batch_idx % 100 == 0 and batch_idx > 0:
            with torch.no_grad():
                # Detach and denormalize frames and reconstructions
                frames_denorm = (frames.cpu() + 1) / 2  # Shape: (batch, channels, time, height, width)
                recon_denorm = (recon.cpu() + 1) / 2    # Shape: (batch, channels, time, height, width)

                # Select the first sequence in the batch
                frames_denorm = frames_denorm[0]  # Shape: (channels, time, height, width)
                recon_denorm = recon_denorm[0]    # Shape: (channels, time, height, width)

                # clip the values to [0, 1] to avoid numerical errors
                frames_denorm = torch.clamp(frames_denorm, 0, 1)
                recon_denorm = torch.clamp(recon_denorm, 0, 1)

                # Rearrange to (time, height, width, channels) for plotting
                frames_denorm = einops.rearrange(frames_denorm, 'c t h w -> t h w c')
                recon_denorm = einops.rearrange(recon_denorm, 'c t h w -> t h w c')

                # Select 5 frames evenly spaced (or fewer if sequence is short)
                t = frames_denorm.shape[0]
                num_frames_to_display = min(5, t)
                indices = np.linspace(0, t-1, num_frames_to_display, dtype=int)

                # Create a figure with 2 rows: original and reconstructed
                fig, axes = plt.subplots(2, num_frames_to_display, figsize=(15, 6))

                for i, idx in enumerate(indices):
                    # Original frame
                    axes[0, i].imshow(frames_denorm[idx])
                    axes[0, i].set_title(f"Orig t={idx}")
                    axes[0, i].axis('off')

                    # Reconstructed frame
                    axes[1, i].imshow(recon_denorm[idx])
                    axes[1, i].set_title(f"Recon t={idx}")
                    axes[1, i].axis('off')

                plt.tight_layout()
                os.makedirs("training_images", exist_ok=True)
                plt.savefig(f"training_images/visualization_step_{batch_idx}.png")
                plt.show()  # Display the plot
        if batch_idx == total_number_of_steps:
            break
    # %%
    torch.save(vae.state_dict(), "vae.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

# %%
