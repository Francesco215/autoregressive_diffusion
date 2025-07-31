#%%
from contextlib import nullcontext
import os
import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import matplotlib.pyplot as plt
from streaming.base.util import clean_stale_shared_memory


from edm2.conv import MPConv
from edm2.plotting import plot_training_dashboard
from edm2.vae.stability import StabilityVAEEncoder
from edm2.cs_dataloading import CsCollate, CsDataset, CsVaeCollate, CsVaeDataset
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.phema import PowerFunctionEMA
from edm2.vae import VAE
import torch._dynamo.config

torch._dynamo.config.cache_size_limit = 100
        

        
def train(device, local_rank=0):
    vae = VAE.from_pretrained('s3://autoregressive-diffusion/saved_models/vae_cs_102354.pt').to(device)
    vae.mean=vae.mean.to(device)
    vae.std=vae.std.to(device)
    unet = UNet(img_resolution=32, # Match your latent resolution
                img_channels=8, # Match your latent channels
                label_dim = 4,
                model_channels=128,
                channel_mult=[1,2,4,4],
                channel_mult_noise=None,
                channel_mult_emb=None,
                num_blocks=2,
                video_attn_resolutions=[4],
                frame_attn_resolutions=[8],
                )

    resume_training=False
    unet_params = sum(p.numel() for p in unet.parameters())
    if resume_training:
        unet=UNet.from_pretrained(f'saved_models/unet_{unet_params//1e6}M.pt')

    unet=unet.to(device)
    if dist.is_available() and dist.is_initialized():
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if local_rank==0:
        print(f"Number of UNet parameters: {unet_params//1e6}M")

    micro_batch_size = 2
    batch_size = 8
    accumulation_steps = batch_size//micro_batch_size
    clip_length = 16
    # training_steps = total_number_of_steps * batch_size
    dataset = CsVaeDataset(clip_size=clip_length, remote='s3://counter-strike-data/vae_40M/', local = f'/data/streaming_dataset/cs_diff', batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsVaeCollate(), pin_memory=True, num_workers=4, shuffle=False)
    steps_per_epoch = len(dataset)//micro_batch_size
    n_epochs = 10
    total_number_of_steps = n_epochs * steps_per_epoch
    # total_number_of_steps = 40_000


    # sigma_data = 0.434
    sigma_data = 1.
    precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to(device)
    loss_fn = EDM2Loss(P_mean=0.9,P_std=1.0, sigma_data=sigma_data, context_noise_reduction=0.1)

    ref_lr = 1e-2
    current_lr = ref_lr
    optimizer = AdamW(precond.parameters(), lr=ref_lr, eps = 1e-4)
    optimizer.zero_grad()
    ema_tracker = PowerFunctionEMA(precond, stds=[0.050, 0.100])
    losses = []

    steps_taken = 0
    if resume_training:
        checkpoint = torch.load(f'saved_models/optimizers_{unet_params//1e6}M.pt', weights_only=False, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ema_tracker.load_state_dict(checkpoint['ema_state_dict'])
        losses = checkpoint['losses']
        current_lr = optimizer.param_groups[0]['lr']
        ref_lr = checkpoint['ref_lr']
        steps_taken = checkpoint['steps_taken']
        print(f"Resuming training from batch {checkpoint['steps_taken']} with loss {losses[-1]:.4f}")

    for epoch in range (n_epochs):
        pbar = tqdm(enumerate(dataloader, start=steps_taken),total=steps_per_epoch) if local_rank==0 else enumerate(dataloader)
        for i, batch in pbar:
            with torch.no_grad():
                means, _ = batch
                means= means.to(device)
                # print(means.shape)
                latents = (means-vae.mean[:,None,None])/vae.std[:,None,None]
                actions = None

            # Calculate loss    
            loss, un_weighted_loss = loss_fn(precond, latents, actions, just_2d=i%4==0)
            # Backpropagation and optimization
            with (nullcontext() if i % accumulation_steps == 0 else unet.no_sync()):
                loss.backward()

            if dist.is_initialized():
                un_weighted_loss = torch.tensor(un_weighted_loss, device=device)
                dist.all_reduce(un_weighted_loss, op=dist.ReduceOp.SUM)
                un_weighted_loss = un_weighted_loss.item()/ dist.get_world_size()
            losses.append(un_weighted_loss)
            if local_rank==0:
                pbar.set_postfix_str(f"Loss: {np.mean(losses[-accumulation_steps:]):.4f}, lr: {current_lr:.6f}, epoch: {epoch+1}")

            if i % accumulation_steps == 0 and i!=0:
                #microbatching
                optimizer.step()
                optimizer.zero_grad()
                ema_tracker.update(cur_nimg= i * batch_size, batch_size=batch_size)

                for g in optimizer.param_groups:
                    current_lr = learning_rate_schedule(i + epoch*steps_per_epoch, ref_lr, total_number_of_steps/500, total_number_of_steps/500)
                    g['lr'] = current_lr

            # Save model checkpoint (optional)
            if i % 500 * accumulation_steps == 0 and i!=0:
                precond.noise_weight.fit_loss_curve()
                if local_rank==0:
                    plot_training_dashboard(
                        save_path=f'images_training/dashboard_step_{i}.png', # Dynamic filename
                        precond=precond,
                        autoencoder=vae,
                        losses_history=losses, # Pass the list of scalar losses
                        current_step=i,
                        micro_batch_size=micro_batch_size,
                        unet_params=unet_params,
                        latents=latents, # Pass the latents from the current batch
                        actions=actions,
                    )

        # if i % (total_number_of_steps//100) == 0 and i!=0:  # save every 10% of epochs
        if local_rank==0:
            os.makedirs("saved_models", exist_ok=True)
            if isinstance(unet, DDP):
                unet.module.save_to_state_dict(f"saved_models/unet_{unet_params//1e6}M.pt")
            else:
                unet.save_to_state_dict(f"saved_models/unet_{unet_params//1e6}M.pt")

            torch.save({
                'steps_taken': i,
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema_tracker.state_dict(),
                'losses': losses,
                'ref_lr': ref_lr
            }, f"saved_models/optimizers_{unet_params//1e6}M.pt")

# %%
        
        
if __name__=="__main__":
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device, local_rank = "cuda", 0

    clean_stale_shared_memory()
    train(device, local_rank)
# %%
