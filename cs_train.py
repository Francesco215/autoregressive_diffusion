#%%
from contextlib import nullcontext
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import matplotlib.pyplot as plt
from streaming.base.util import clean_stale_shared_memory


from edm2.vae import VAE
from edm2.cs_dataloading import CsCollate, CsDataset, CsVaeCollate, CsVaeDataset
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.phema import PowerFunctionEMA
import torch._dynamo.config

torch._dynamo.config.cache_size_limit = 100
        
if __name__=="__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)

    clean_stale_shared_memory()
    

    unet = UNet(img_resolution=32, # Match your latent resolution
                img_channels=8, # Match your latent channels
                label_dim = 4,
                model_channels=128,
                channel_mult=[1,2,4,4],
                channel_mult_noise=None,
                channel_mult_emb=None,
                num_blocks=2,
                attn_resolutions=[8,4]
                )
    resume_training=False
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of UNet parameters: {unet_params//1e6}M")
    if resume_training:
        unet=UNet.from_pretrained(f'saved_models/unet_{unet_params//1e6}M.pt')
    unet = DDP(unet.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    micro_batch_size = 1
    batch_size = 8
    accumulation_steps = batch_size//micro_batch_size
    clip_length = 8
    # training_steps = total_number_of_steps * batch_size
    dataset = CsVaeDataset(clip_size=clip_length, remote='s3://counter-strike-data/dataset_compressed/', local = f'/tmp/streaming_dataset/cs_vae', batch_size=micro_batch_size, shuffle=False, cache_limit = '5gb')
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsVaeCollate(), pin_memory=True, num_workers=24, shuffle=False)
    steps_per_epoch = len(dataset)//micro_batch_size
    n_epochs = 10
    total_number_of_steps = n_epochs * steps_per_epoch
    # total_number_of_steps = 40_000


    # sigma_data = 0.434
    sigma_data = 1.
    precond = Precond(unet, use_fp16=True, sigma_data=sigma_data).to(device)
    loss_fn = EDM2Loss(P_mean=0.3,P_std=2., sigma_data=sigma_data, context_noise_reduction=0.5)

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

    #%%
    for epoch in range (n_epochs):
        pbar = tqdm(enumerate(dataloader, start=steps_taken),total=steps_per_epoch)
        for i, batch in pbar:
            with torch.no_grad():
                latents, _, _ = batch
                latents = latents.to(device)/1.3
                actions = None
                
                    
            # Calculate loss    
            loss, un_weighted_loss = loss_fn(precond, latents, actions)
            losses.append(un_weighted_loss)
            # Backpropagation and optimization
            with (precond.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext()):
                loss.backward()
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
            if i % 50 * accumulation_steps == 0 and i!=0 and local_rank==0:
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

        # if i % (total_number_of_steps//100) == 0 and i!=0:  # save every 10% of epochs
        if local_rank==0:
            unet.save_to_state_dict(f"saved_models/unet_{unet_params//1e6}M.pt")
            torch.save({
                'steps_taken': i,
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema_tracker.state_dict(),
                'losses': losses,
                'ref_lr': ref_lr
            }, f"saved_models/optimizers_{unet_params//1e6}M.pt")

# %%
