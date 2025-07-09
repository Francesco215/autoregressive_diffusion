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

# torch.autograd.set_detect_anomaly(True)
if __name__=="__main__":
    device = "cuda"
    original_env = "LunarLander-v3"

    batch_size = 16
    micro_batch_size = 4
    clip_length = 32 
    
    # Hyperparameters
    latent_channels = 8
    n_res_blocks = 3
    channels = [3, 64, 256, 512, latent_channels]

    # Initialize models
    vae = VAE(channels = channels, n_res_blocks=n_res_blocks, spatial_compressions=[1,2,2,2], time_compressions=[1,2,2,1], logvar_mode=0.1).to(device)
    vae = torch.compile(vae)
    # Example instantiation
    #%%
    
    dataset = CsDataset(clip_size=clip_length, remote='s3://counter-strike-data/original/', local = '/tmp/streaming_dataset/cs_vae',batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsCollate(clip_length), num_workers=8, shuffle=False)
    total_number_of_steps = len(dataloader)//micro_batch_size

    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"Number of vae parameters: {vae_params//1e3}K")
    # sigma_data = 0.434
    sigma_data = 1.

    # Define optimizers
    base_lr = 3e-4
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


    recon_losses, kl_losses, losses = [], [], []

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
            main_loss = recon_loss + kl_loss*1e-4 
            main_loss.backward()

            if batch_idx % (batch_size//micro_batch_size) == 0 and batch_idx!=0:
                nn.utils.clip_grad_norm_(vae.parameters(), 1)
                optimizer_vae.step()
                scheduler_vae.step()
                optimizer_vae.zero_grad()


            pbar.set_postfix_str(f"recon loss: {recon_loss.item():.4f}, KL loss: {kl_loss.item():.4f}, current_lr: {optimizer_vae.param_groups[0]['lr']:.4f}")
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

            
            if batch_idx % 1000 == 0 and batch_idx > 0:
                fig = plt.figure(figsize=(15, 8))

                # Top section: 2 rows for original and reconstructed frames
                gs_top = plt.GridSpec(2, 5, figure=fig, top=0.95, bottom=0.55, left=0.1, right=0.9)
                orig_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
                recon_axes = [fig.add_subplot(gs_top[1, i]) for i in range(5)]

                # Bottom section: 1x2 for loss plots
                gs_bottom = plt.GridSpec(1, 2, figure=fig, top=0.45, bottom=0.1, left=0.1, right=0.9, hspace=0.4)
                loss_axes = [
                    fig.add_subplot(gs_bottom[0, 0]),
                    fig.add_subplot(gs_bottom[0, 1])
                ]

                # Frame visualization
                with torch.no_grad():
                    frames_denorm = (frames.cpu() + 1) / 2
                    recon_denorm = (recon.cpu() + 1) / 2

                    frames_denorm = torch.clamp(frames_denorm[0], 0, 1)  # (c, t, h, w)
                    recon_denorm = torch.clamp(recon_denorm[0], 0, 1)

                    frames_denorm = einops.rearrange(frames_denorm, 'c t h w -> t h w c')
                    recon_denorm = einops.rearrange(recon_denorm, 'c t h w -> t h w c')

                    t = frames_denorm.shape[0]
                    indices = np.linspace(0, t - 1, 5, dtype=int)

                    for i, idx in enumerate(indices):
                        orig_axes[i].imshow(frames_denorm[idx])
                        orig_axes[i].set_title(f"Orig t={idx}")
                        orig_axes[i].axis('off')

                        recon_axes[i].imshow(recon_denorm[idx])
                        recon_axes[i].set_title(f"Recon t={idx}")
                        recon_axes[i].axis('off')

                # Plot reconstruction loss
                loss_axes[0].plot(recon_losses, label="Recon Loss", color="blue")
                loss_axes[0].set_title("Reconstruction Loss")
                loss_axes[0].set_yscale("log")
                loss_axes[0].set_xscale("log")
                loss_axes[0].set_xlabel("Steps")
                loss_axes[0].set_ylabel("Loss")
                loss_axes[0].grid(True)

                # Plot KL loss
                loss_axes[1].plot(kl_losses, label="KL Loss", color="red")
                loss_axes[1].set_title("KL Loss")
                loss_axes[1].set_yscale("log")
                loss_axes[1].set_xscale("log")
                loss_axes[1].set_xlabel("Steps")
                loss_axes[1].set_ylabel("Loss")
                loss_axes[1].grid(True)

                plt.tight_layout()
                os.makedirs("images_training", exist_ok=True)
                plt.savefig(f"images_training/combined_step_cs_{batch_idx}.png")
                plt.close()

            if batch_idx % (total_number_of_steps // 10) == 0 and batch_idx != 0:
                os.makedirs("saved_models", exist_ok=True)
                vae.save_to_state_dict(f'saved_models/vae_cs_{batch_idx}.pt')

    print("Finished Training")
    # %%