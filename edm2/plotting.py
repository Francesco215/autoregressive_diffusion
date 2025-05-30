from torch import distributed as dist
from edm2.networks_edm2 import Precond
from .sampler import edm_sampler_with_mse
import matplotlib.pyplot as plt
import numpy as np
import torch
import einops
from matplotlib.colors import LogNorm # Make sure LogNorm is imported
from tqdm import tqdm # If not already imported where this function is defined
import os

# Assuming edm_sampler_with_mse and latents_to_frames are defined/imported
# from edm2.sampler import edm_sampler_with_mse # Or wherever it is
# from edm2.gym_dataloader import latents_to_frames # Or wherever it is

# Assuming MultiNoiseLoss class is defined as provided by the user
# from your_module import MultiNoiseLoss # Or wherever it is

@torch.no_grad()
def plot_training_dashboard(
    save_path,
    precond:Precond, # precond object which should have precond.noise_weight of type MultiNoiseLoss
    autoencoder,
    losses_history,
    current_step,
    micro_batch_size,
    unet_params,
    latents, 
    actions,
    guidance=1,
    ):
    """
    Generates and saves a consolidated 2x2 plot dashboard for training monitoring.
    Uses the plotting logic from MultiNoiseLoss directly on a subplot.
    Uses the frame generation logic directly from sampler_training_callback.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Dashboard - Step {current_step}', fontsize=16)

    # --- Plot 1: Loss vs Sigma (Top-Left) ---
    # (Code remains the same as the previous corrected version)
    ax1 = axes[0, 0]
    noise_weight_module = precond.noise_weight # Get the MultiNoiseLoss module
    if hasattr(noise_weight_module, 'sigmas') and hasattr(noise_weight_module, 'losses') and noise_weight_module.sigmas.numel() > 0:
        sigmas_cpu = noise_weight_module.sigmas.cpu()
        losses_cpu = noise_weight_module.losses.cpu()
        positions_cpu = noise_weight_module.positions.cpu()
        scatter = ax1.scatter(
            sigmas_cpu, losses_cpu, c=positions_cpu, cmap='viridis', norm=LogNorm(),
            alpha=0.8, label='Data Points', s=1.0
        )
        fig.colorbar(scatter, ax=ax1, label='Position', fraction=0.046, pad=0.04)
        num_points = 200
        min_sigma_data = sigmas_cpu.min().item() if sigmas_cpu.numel() > 0 else 1e-2
        max_sigma_data = sigmas_cpu.max().item() if sigmas_cpu.numel() > 0 else 1e2
        min_sigma_plot = max(1e-3, min_sigma_data)
        max_sigma_plot = min(1e3, max_sigma_data)
        if min_sigma_plot >= max_sigma_plot:
             min_sigma_plot = 1e-2
             max_sigma_plot = 1e2
        sigma_values_plot = torch.logspace(np.log10(min_sigma_plot), np.log10(max_sigma_plot), num_points)
        if hasattr(noise_weight_module, 'calculate_mean_loss'):
            mean_loss_plot = noise_weight_module.calculate_mean_loss(sigma_values_plot.to(noise_weight_module.sigmas.device))
            ax1.plot(sigma_values_plot.cpu(), mean_loss_plot.cpu(), label='Best Fit', color='red', linewidth=2)
        else:
             print("Warning: calculate_mean_loss not found on noise_weight_module.")
        ax1.set_xscale('log')
        ax1.set_xlabel('σ (sigma)')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.set_title('Loss vs. σ')
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.5)
    else:
        ax1.text(0.5, 0.5, 'Noise vs Loss data not available', horizontalalignment='center', verticalalignment='center')
        ax1.set_title('Loss vs. σ')
        ax1.grid(True)


    # --- Plot 2: Training Loss History (Top-Right) ---
    # (Code remains the same as the previous corrected version)
    ax2 = axes[0, 1]
    if losses_history:
        scalar_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in losses_history]
        n_clips = np.linspace(0, current_step * micro_batch_size, len(scalar_losses))
        ax2.plot(n_clips, scalar_losses, label='Raw Loss', color='blue', alpha=0.3)
        if len(scalar_losses) >= 100:
            moving_avg = np.convolve(scalar_losses, np.ones(100) / 100, mode='valid')
            n_images_avg_start_idx = len(scalar_losses) - len(moving_avg)
            n_images_avg = n_clips[n_images_avg_start_idx:]
            ax2.plot(n_images_avg, moving_avg, label='Moving Avg (100 steps)', color='blue', alpha=1)
        ax2.set_xscale('log')
        ax2.set_xlabel('N Frames Seen')
        ax2.set_ylabel('Loss')
        ax2.set_yscale('log')
        ax2.set_title(f'Training Loss ({unet_params // 1e6:.1f}M params)')
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.5)
    else:
         ax2.text(0.5, 0.5, 'No loss history yet', ha='center', va='center')
         ax2.set_title(f'Training Loss ({unet_params // 1e6:.1f}M params)')
         ax2.grid(True)

    # --- Plot 3 & 4 Prep: Sampling (Requires eval mode) ---
    precond.eval() # Set model to evaluation mode for sampling

    # Select subset of latents for visualization
    # Use a clone here to avoid modifying the original batch_latents if it's used later

    # --- Plot 3: Denoising MSE (Bottom-Left) ---
    # (Code remains the same as the previous corrected version)
    # Uses latents_viz_orig
    ax3 = axes[1, 0]
    latents = latents[:,:7]
    # latents = batch["latents"][start:start+num_samples].to(device)
    # text_embeddings = batch["text_embeddings"][start:start+num_samples].to(device)
    context = latents[:, :-1]  # First frames (context)
    target = latents[:, -1:]    # Last frame (ground truth)
    precond.eval()
    sigma = torch.ones(context.shape[:2], device=latents.device) * 0.05
    conditioning = None if actions is None else actions[:,:context.shape[1]] 
    _, cache = precond.forward(context, sigma, conditioning, update_cache=True)

    # Run sampler with sigma_max=0.5 for initial noise level
    conditioning = None if actions is None else actions[:,context.shape[1]:context.shape[1]+1]
    sigma_max_val = 3.0   # <- must match the call you make just below
    sigma_min = 0.8
    rho_val        = 2.0
    num_steps_val  = 32
    _, mse_steps, mse_pred_values, _ = edm_sampler_with_mse(net=precond, cache=cache, target=target, sigma_max=sigma_max_val, sigma_min=sigma_min, num_steps=num_steps_val, conditioning=conditioning, rho = rho_val, guidance = guidance, S_churn=20, S_noise=1,
    )

    # Plot results
    # is it possible to plot above the corresponding times ot the timesteps
    #they are determined like this #!wrong code below!#

    ax3.plot(mse_steps, marker='o', linestyle='-', label="MSE")
    ax3.plot(mse_pred_values, marker='o', linestyle='-', label="MSE (Predicted)")
    ax3.set_xlabel("Denoising Step")
    ax3.set_ylabel("MSE")
    ax3.set_yscale("log")
    ax3.set_title("Denoising Progress (Lower is Better)")
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    # ax3.legend()
    step_idx  = torch.arange(num_steps_val, device=latents.device)
    sigmas_ts = (sigma_max_val ** (1/rho_val)
                + step_idx / (num_steps_val - 1)
                * (sigma_min ** (1/rho_val) - sigma_max_val ** (1/rho_val))
                ) ** rho_val
    sigmas_ts = torch.cat([sigmas_ts, sigmas_ts.new_zeros(1)])  # final σ = 0
    sigma_lbls = [f'{s.item():.2f}' for s in sigmas_ts]         # or use f'{s:.1e}'

    ax3_top = ax3.secondary_xaxis('top')          # use ax3.twiny() for MPL < 3.4
    ax3_top.set_xlim(ax3.get_xlim())              # keep identical limits
    ax3_top.set_xticks(range(len(sigmas_ts)))
    ax3_top.set_xticklabels(sigma_lbls, rotation=45, ha='left', fontsize=8)
    ax3_top.set_xlabel("σ-timestep")

    # --- Plot 4: Generated Frames (Bottom-Right) ---
    # Replicate the *exact* logic from sampler_training_callback
    ax4 = axes[1, 1]
    # for _ in tqdm(range(6)):
    #     actions = None if actions is None else torch.randint(0,3,(latents.shape[0],1), device=latents.device)
    #     x, _, _, cache= edm_sampler_with_mse(precond, cache=cache, conditioning = actions, sigma_max = 80, sigma_min=sigma_min, num_steps=16, rho=2, guidance=guidance, S_churn=0.)
    #     context = torch.cat((context,x),dim=1)
    
    # # context = einops.rearrange(context, 'b t (c hs ws) h w -> b t c (h hs) (w ws) ', hs=2, ws=2)
    # frames = autoencoder.latents_to_frames(context)

    # x = einops.rearrange(frames, 'b (t1 t2) h w c -> b (t1 h) (t2 w) c', t2=8)
    # #set high resolution
    # ax4.imshow(x[0])
    # ax4.axis('off')



    # --- Final Steps ---
    precond.train() # Set model back to training mode

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    os.makedirs("images_training", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved to {save_path}")
    plt.close(fig) # Close the figure to free memory