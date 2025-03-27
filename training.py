#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from streaming import StreamingDataset
import einops

from diffusers import AutoencoderKLMochi
from tqdm import tqdm
import matplotlib.pyplot as plt



from edm2.gym_dataloader import GymDataGenerator, gym_collate_function, frames_to_latents
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.mars import MARS
from edm2.phema import PowerFunctionEMA

torch._dynamo.config.recompile_limit = 100

autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float16)
vae_mean, vae_std = torch.tensor(autoencoder.config.latents_mean)[:, None, None], torch.tensor(autoencoder.config.latents_std)[:, None, None]
def collate_function(batch):
    # batch is a list of dicts with keys 'mean', 'logvar', 'actions'
    # we want to return a tuple of tensors (means, logvars, actions)
    means = torch.tensor(np.array([b['mean'] for b in batch]))
    logvars = torch.tensor(np.array([b['logvar'] for b in batch]), dtype = torch.float32)
    actions = torch.tensor(np.array([b['action'] for b in batch]), dtype = torch.float32)

    # now i sample a frame 
    frames = means + torch.exp(0.5 * logvars) * torch.randn_like(means)
    frames = einops.rearrange(frames, 'b c t h w -> b t c h w')
    frames = (frames - vae_mean)/vae_std
    # frames = frames[:, :16]
    return frames*0.7, actions
        
#%%
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_env = "LunarLander-v3"
    model_id="stabilityai/stable-diffusion-2-1"

    # autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).eval().requires_grad_(False)

    # autoencoder = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float32).to("cuda")
    # autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")
    latent_channels = 12

    unet = UNet(img_resolution=32, # Match your latent resolution
                img_channels=latent_channels, # Match your latent channels
                label_dim = 4,
                model_channels=128,
                channel_mult=[1,2,4,4],
                channel_mult_noise=None,
                channel_mult_emb=None,
                num_blocks=3,
                attn_resolutions=[8,4]
                )

    micro_batch_size = 1
    batch_size = 8
    accumulation_steps = batch_size//micro_batch_size
    state_size = 64 
    # training_steps = total_number_of_steps * batch_size
    dataset = StreamingDataset(remote='s3://counter-strike-data/dataset_small/',batch_size=micro_batch_size)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=collate_function, num_workers=16)
    steps_per_epoch = len(dataset)//micro_batch_size
    n_epochs = 10
    total_number_of_steps = n_epochs * steps_per_epoch
    # total_number_of_steps = 40_000


    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of UNet parameters: {unet_params//1e6}M")
    # sigma_data = 0.434
    sigma_data = 1.
    precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to(device)
    loss_fn = EDM2Loss(P_mean=0.3,P_std=2., sigma_data=sigma_data, context_noise_reduction=0.5)

    ref_lr = 2e-3
    current_lr = ref_lr
    optimizer = MARS(precond.parameters(), lr=ref_lr, eps = 1e-4)
    optimizer.zero_grad()
    ema_tracker = PowerFunctionEMA(precond, stds=[0.050, 0.100])
    losses = []

    resume_training_run = None
    # resume_training_run = 'lunar_lander_68.0M.pt'
    steps_taken = 0
    if resume_training_run is not None:
        print(f"Resuming training from {resume_training_run}")
        checkpoint = torch.load(resume_training_run, weights_only=False, map_location='cuda')
        precond.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ema_tracker.load_state_dict(checkpoint['ema_state_dict'])
        losses = checkpoint['losses']
        print(f"Resuming training from batch {checkpoint['batch']} with loss {losses[-1]:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        ref_lr = checkpoint['ref_lr']
        steps_taken = checkpoint['batch']

    #%%
    for epoch in range (n_epochs):
        pbar = tqdm(enumerate(dataloader, start=steps_taken),total=steps_per_epoch)
        for i, batch in pbar:
            with torch.no_grad():
                frames, actions = batch
                latents = frames.to(device)
                actions = None #if i%4==0 else actions.to(device)

            # Calculate loss    
            loss, un_weighted_loss = loss_fn(precond, latents, actions)
            losses.append(un_weighted_loss)
            # Backpropagation and optimization
            loss.backward()
            pbar.set_postfix_str(f"Loss: {np.mean(losses[-accumulation_steps:]):.4f}, lr: {current_lr:.6f}, epoch: {epoch+1}")

            if i % accumulation_steps == 0 and i!=0:
                #microbatching
                optimizer.step()
                optimizer.zero_grad()
                ema_tracker.update(cur_nimg= i * batch_size, batch_size=batch_size)

                for g in optimizer.param_groups:
                    current_lr = learning_rate_schedule(i + epoch*steps_per_epoch, ref_lr, total_number_of_steps/50, total_number_of_steps/50)
                    g['lr'] = current_lr

            # Save model checkpoint (optional)
            if i % 50 * accumulation_steps == 0 and i!=0:
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

        # if i % (total_number_of_steps//100) == 0 and i!=0:  # save every 10% of epochs
        torch.save({
            'batch': i,
            'model_state_dict': precond.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_state_dict': ema_tracker.state_dict(),
            'losses': losses,
            'ref_lr': ref_lr
        }, f"lunar_lander_{unet_params//1e6}M.pt")

# %%
