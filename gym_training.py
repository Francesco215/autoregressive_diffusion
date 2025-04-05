#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import matplotlib.pyplot as plt



from edm2.vae import VAE    
from edm2.gym_dataloader import GymDataGenerator, gym_collate_function, frames_to_latents
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.mars import MARS
from edm2.phema import PowerFunctionEMA

# torch._dynamo.config.recompile_limit = 100
# torch.autograd.set_detect_anomaly(True)
#%%
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"
    model_id="stabilityai/stable-diffusion-2-1"

    autoencoder = VAE.from_pretrained("saved_models/vae_4000.pt").to(device).requires_grad_(False)

    resume_training = True
    unet = UNet(img_resolution=256//autoencoder.spatial_compression, # Match your latent resolution
                img_channels=autoencoder.latent_channels, # Match your latent channels
                label_dim = 4, #this should be equal to the action space of the gym environment
                model_channels=16,
                channel_mult=[1,2,2,4],
                channel_mult_noise=None,
                channel_mult_emb=None,
                num_blocks=3,
                attn_resolutions=[]
                )

    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of UNet parameters: {unet_params//1e6}M")
    if resume_training:
        unet=UNet.from_pretrained(f'saved_models/unet_{unet_params//1e6}M.pt')

    micro_batch_size = 8
    batch_size = micro_batch_size
    accumulation_steps = batch_size//micro_batch_size
    state_size = 32 
    total_number_of_steps = 40_000
    training_steps = total_number_of_steps * batch_size
    dataset = GymDataGenerator(state_size, original_env, total_number_of_steps, autoencoder_time_compression = autoencoder.time_compression)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=gym_collate_function, num_workers=micro_batch_size, prefetch_factor=4)

    # sigma_data = 0.434
    sigma_data = 1.
    precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to(device)
    loss_fn = EDM2Loss(P_mean=0.3,P_std=2., sigma_data=sigma_data, context_noise_reduction=0.5)

    ref_lr = 1e-2
    current_lr = ref_lr
    optimizer = AdamW(precond.parameters(), lr=ref_lr, eps = 1e-8)
    optimizer.zero_grad()
    ema_tracker = PowerFunctionEMA(precond, stds=[0.050, 0.100])
    losses, steps_taken = [], 0

    if resume_training:
        checkpoint = torch.load(f'saved_models/optimizers_{unet_params//1e6}M.pt', weights_only=False, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ema_tracker.load_state_dict(checkpoint['ema_state_dict'])
        losses = checkpoint['losses']
        current_lr = optimizer.param_groups[0]['lr']
        ref_lr = checkpoint['ref_lr']
        steps_taken = checkpoint['steps_taken']
        print(f"Resuming training from batch {checkpoint['steps_taken']} with loss {losses[-1]:.4f}")

    #%%
    pbar = tqdm(enumerate(dataloader, start=steps_taken),total=total_number_of_steps)
    for i, batch in pbar:
        with torch.no_grad():
            frames, actions, _ = batch
            frames = torch.tensor(frames, device=device)
            actions = torch.tensor(actions, device = device)
            latents = frames_to_latents(autoencoder, frames)/1.2

        # Calculate loss    
        loss, un_weighted_loss = loss_fn(precond, latents, actions)
        losses.append(un_weighted_loss)
        # Backpropagation and optimization
        loss.backward()
        pbar.set_postfix_str(f"Loss: {np.mean(losses[-accumulation_steps:]):.4f}, lr: {current_lr:.6f}")

        if i % accumulation_steps == 0 and i!=0:
            #microbatching
            
            clip_grad_norm_(precond.parameters(), .1)
            optimizer.step()
            optimizer.zero_grad()
            ema_tracker.update(cur_nimg= i * batch_size, batch_size=batch_size)

            for g in optimizer.param_groups:
                current_lr = learning_rate_schedule(i, ref_lr, total_number_of_steps/50, total_number_of_steps/50)
                g['lr'] = current_lr

        # Save model checkpoint (optional)
        if i % 200 * accumulation_steps == 0 and i!=0:
            precond.noise_weight.fit_loss_curve()
            precond.noise_weight.plot('images_training/plot.png')
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
            plt.savefig('images_training/losses.png')
            plt.show()
            plt.close()
            ulw=True

        if i % (total_number_of_steps//40) == 0 and i!=0:  # save every 10% of epochs
            unet.save_to_state_dict(f"saved_models/unet_{unet_params//1e6}M.pt")
            torch.save({
                'steps_taken': i,
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema_tracker.state_dict(),
                'losses': losses,
                'ref_lr': ref_lr
            }, f"saved_models/optimizers_{unet_params//1e6}M.pt")

        if i == total_number_of_steps:
            break
# %%
