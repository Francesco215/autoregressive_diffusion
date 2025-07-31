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
from edm2.vae import VAE

os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/mnt/mnemo9/mpelus/experiments/autoregressive_diffusion/.torchinductor_cache'

def calculate_latent_std_and_save(model_path, save_path, dataset_config=None, num_samples=10000):
    """
    Load a VAE model, run encoder over dataset to collect specified number of samples,
    calculate latent std, update the model with the std, and save it.
    
    Args:
        model_path (str): Path to the trained VAE model (can be S3 path)
        save_path (str): Path to save the updated model with std (can be S3 path)
        dataset_config (dict): Dataset configuration, uses defaults if None
        num_samples (int): Number of latent samples to collect (default: 10000)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default dataset configuration
    if dataset_config is None:
        dataset_config = {
            'clip_length': 32,
            'micro_batch_size': 1,
            'remote': 's3://counter-strike-data/original/',
            'local': '/mnt/mnemo9/mpelus/experiments/autoregressive_diffusion/streaming_dataset/cs_vae',
            'cache_limit': '50gb'
        }
    
    print(f"Loading VAE model from: {model_path}")
    # Load the trained VAE model
    vae = VAE.from_pretrained(model_path).to(device)
    vae.eval()  # Set to evaluation mode
    
    print(f"VAE loaded. Current std: {vae.std}")
    print(f"VAE parameters: {vae.n_params//1e6:.1f}M")
    
    # Setup dataset
    print("Setting up dataset...")
    dataset = CsDataset(
        clip_size=dataset_config['clip_length'], 
        remote=dataset_config['remote'], 
        local=dataset_config['local'],
        batch_size=dataset_config['micro_batch_size'], 
        shuffle=True,  # Dataset handles its own shuffling
        cache_limit=dataset_config['cache_limit']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=dataset_config['micro_batch_size'], 
        collate_fn=CsCollate(dataset_config['clip_length']), 
        num_workers=8
        # No shuffle parameter for IterableDataset - shuffling handled by dataset itself
    )
    
    print(f"Dataset loaded. Total batches: {len(dataloader)}")
    print(f"Target samples: {num_samples}")
    
    # Collect latent representations
    print(f"Collecting {num_samples} latent samples...")
    all_latents = []
    samples_collected = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Collecting samples")
        
        for batch_idx, micro_batch in enumerate(dataloader):
            try:
                frames, _ = micro_batch  # Ignore actions and reward
                frames = frames.float() / 127.5 - 1  # Normalize to [-1, 1]
                frames = einops.rearrange(frames, 'b t h w c -> b c t h w').to(device)
                
                # Encode frames to get latent mean
                latent_mean, _ = vae.encode(frames)
                
                # Flatten latents for std calculation: [B, C, T, H, W] -> [B*T*H*W, C]
                latent_flat = einops.rearrange(latent_mean, 'b c t h w -> (b t h w) c')
                
                # Calculate how many samples this batch contributes
                batch_samples = latent_flat.shape[0]
                
                # If this batch would exceed our target, subsample it
                if samples_collected + batch_samples > num_samples:
                    remaining_samples = num_samples - samples_collected
                    # Randomly subsample the required number
                    indices = torch.randperm(batch_samples)[:remaining_samples]
                    latent_flat = latent_flat[indices]
                    batch_samples = remaining_samples
                
                all_latents.append(latent_flat.cpu())
                samples_collected += batch_samples
                
                # Update progress bar
                pbar.set_postfix_str(f"Samples: {samples_collected}/{num_samples}")
                pbar.update(1)
                
                # Break if we've collected enough samples
                if samples_collected >= num_samples:
                    break
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        pbar.close()
    
    # Concatenate all latents
    print("Calculating statistics...")
    all_latents = torch.cat(all_latents, dim=0)  # Shape: [N_samples, C]
    print(f"Final sample count: {all_latents.shape[0]} samples across {all_latents.shape[1]} channels")
    
    # Calculate statistics
    # Overall statistics (across all channels and samples)
    overall_mean = torch.mean(all_latents).item()
    overall_std = torch.std(all_latents).item()
    
    # Per-channel statistics
    channel_means = torch.mean(all_latents, dim=0)  # Shape: [C]
    channel_stds = torch.std(all_latents, dim=0)    # Shape: [C]
    
    # Mean across channels (this is what we'll use for the VAE)
    mean_channel_mean = torch.mean(channel_means).item()
    mean_channel_std = torch.mean(channel_stds).item()
    
    print(f"\n=== STATISTICS ===")
    print(f"Overall mean: {overall_mean:.6f}")
    print(f"Overall std: {overall_std:.6f}")
    print(f"Mean of per-channel means: {mean_channel_mean:.6f}")
    print(f"Mean of per-channel stds: {mean_channel_std:.6f}")
    
    print(f"\nPer-channel means: {[f'{x:.6f}' for x in channel_means.tolist()]}")
    print(f"Per-channel stds:  {[f'{x:.6f}' for x in channel_stds.tolist()]}")
    
    # Create new VAE with the calculated std and mean
    print(f"\nCreating updated VAE with mean={mean_channel_mean:.6f}, std={mean_channel_std:.6f}...")
    
    # Get the original kwargs and update with std (mean will be set as attribute)
    updated_kwargs = vae.kwargs.copy()
    updated_kwargs['std'] = mean_channel_std
    
    # Create new VAE instance with updated kwargs
    updated_vae = VAE(**updated_kwargs)
    
    # Copy the trained state dict
    updated_vae.load_state_dict(vae.state_dict())
    
    # Set the mean as an attribute (since it's not a constructor parameter)
    updated_vae.mean = mean_channel_mean
    
    # Verify the std and mean were set correctly
    print(f"Updated VAE mean: {updated_vae.mean}")
    print(f"Updated VAE std: {updated_vae.std}")
    
    # Save the updated model
    print(f"Saving updated VAE to: {save_path}")
    updated_vae.save_to_state_dict(save_path)
    
    print("Done! VAE saved with calculated latent standard deviation.")
    
    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'mean_channel_mean': mean_channel_mean,
        'mean_channel_std': mean_channel_std,
        'channel_means': channel_means.tolist(),
        'channel_stds': channel_stds.tolist(),
        'samples_used': all_latents.shape[0]
    }

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "vae_cs_102354.pt"  
    SAVE_PATH = "s3://autoregressive-diffusion/saved_models/vae_cs_102354.pt" 
    
    # Optional: customize dataset configuration
    dataset_config = {
        'clip_length': 32,
        'micro_batch_size': 5,  
        'remote': 's3://counter-strike-data/original/',
        'local': '/mnt/mnemo9/mpelus/experiments/autoregressive_diffusion/streaming_dataset/cs_vae',
        'cache_limit': '50gb'
    }
    
    # Run the calculation and save
    try:
        results = calculate_latent_std_and_save(
            MODEL_PATH, 
            SAVE_PATH, 
            dataset_config,
            num_samples=10000  # Much faster - just 10k samples
        )
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Samples used: {results['samples_used']}")
        print(f"Overall mean: {results['overall_mean']:.6f}")
        print(f"Overall std: {results['overall_std']:.6f}")
        print(f"Mean of per-channel means (used for VAE): {results['mean_channel_mean']:.6f}")
        print(f"Mean of per-channel stds (used for VAE): {results['mean_channel_std']:.6f}")
        print(f"Per-channel means: {results['channel_means']}")
        print(f"Per-channel stds: {results['channel_stds']}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise