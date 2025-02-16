#%%
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

import einops
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL


from edm2.gym_dataloader import GymDataGenerator, gym_collate_function, frames_to_latents
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.loss_weight import MultiNoiseLoss
from edm2.mars import MARS
from edm2.phema import PowerFunctionEMA

torch._dynamo.config.recompile_limit = 100
#%%
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"
    model_id="stabilityai/stable-diffusion-2-1"

    autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).requires_grad_(False)
    latent_channels = autoencoder.config.latent_channels

    unet = UNet(img_resolution=32, # Match your latent resolution
                img_channels=latent_channels, # Match your latent channels
                label_dim = 4,
                model_channels=64,
                channel_mult=[1,2,2,4],
                channel_mult_noise=None,
                channel_mult_emb=None,
                num_blocks=3,
                attn_resolutions=[8,4]
                )

    micro_batch_size = 16
    batch_size = 16
    accumulation_steps = batch_size//micro_batch_size
    state_size = 16 
    total_number_of_steps = 20_000
    training_steps = total_number_of_steps * batch_size
    dataset = GymDataGenerator(state_size, original_env, training_steps)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=gym_collate_function, num_workers=micro_batch_size)

    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of UNet parameters: {unet_params//1e6}M")
    # sigma_data = 0.434
    sigma_data = 1.
    precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to(device)
    loss_fn = EDM2Loss(P_mean=0.3,P_std=2., sigma_data=sigma_data, noise_weight=MultiNoiseLoss(), context_noise_reduction=0.5)

    ref_lr = 1e-4
    current_lr = ref_lr
    optimizer = MARS(precond.parameters(), lr=ref_lr, eps = 1e-4)
    optimizer.zero_grad()

    ema_tracker = PowerFunctionEMA(precond, stds=[0.050, 0.100])
    losses = []

    resume_training_run = 'lunar_lander_68.0M_trained.pt'
    # resume_training_run = None

    if resume_training_run is not None:
        checkpoint = torch.load(resume_training_run, weights_only=False, map_location='cuda')
        precond.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ema_tracker.load_state_dict(checkpoint['ema_state_dict'])
        losses = checkpoint['losses']
        print(f"Resuming training from batch {checkpoint['batch']} with loss {losses[-1]:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        ref_lr = checkpoint['ref_lr']
    #%%
    ulw=False
    pbar = tqdm(enumerate(dataloader),total=total_number_of_steps)
    for i, batch in pbar:
        frames, actions, reward = batch
        frames = frames.to(device)
        actions = None if i%4==0 else actions.to(device)
        # action_emb = diffusion.action_embedder(action)

        latents = frames_to_latents(autoencoder, frames)

        # Calculate loss    
        loss, un_weighted_loss = loss_fn(precond, latents, actions, use_loss_weight=ulw)
        losses.append(un_weighted_loss)
        # Backpropagation and optimization
        loss.backward()
        pbar.set_postfix_str(f"Loss: {np.mean(losses[-accumulation_steps:]):.4f}, lr: {current_lr:.6f}")

        if i % accumulation_steps == 0 and i!=0:
            #microbatching
            optimizer.step()
            optimizer.zero_grad()
            ema_tracker.update(cur_nimg= i * batch_size, batch_size=batch_size)

            for g in optimizer.param_groups:
                current_lr = learning_rate_schedule(i, ref_lr, total_number_of_steps/50, total_number_of_steps/50)
                g['lr'] = current_lr

        # Save model checkpoint (optional)
        if i % 50 * accumulation_steps == 0 and i!=0:
            loss_fn.noise_weight.fit_loss_curve()
            loss_fn.noise_weight.plot('plot.png')
            n_clips = np.linspace(0, i * micro_batch_size, len(losses))
            plt.plot(n_clips, losses, label='Loss', color='blue', alpha=0.5)
            if len(losses) >= 100:
                moving_avg = np.convolve(losses, np.ones(100) / 100, mode='valid')
                n_images_avg = np.linspace(0, i * micro_batch_size, len(moving_avg))
                plt.plot(n_images_avg, moving_avg, label='Moving Average', color='blue', alpha=1)
            plt.xscale('log')
            plt.xlabel('n images')
            plt.ylabel('loss')
            plt.yscale('log')
            plt.title(f'Losses with {unet_params} parameters')
            plt.savefig('losses.png')
            plt.show()
            plt.close()
            ulw=True

        if i % (total_number_of_steps//100) == 0 and i!=0:  # save every 10% of epochs
            torch.save({
                'batch': i,
                'model_state_dict': precond.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema_tracker.state_dict(),
                'losses': losses,
                'ref_lr': ref_lr
            }, f"lunar_lander_{unet_params//1e6}M.pt")

        if i == total_number_of_steps:
            break