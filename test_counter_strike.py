#%%
# Import necessary libraries
from PIL import Image
import einops
import numpy as np
import torch
from datasets import load_dataset
from diffusers import AutoencoderKLMochi
import os
import h5py
import cv2

from typing import Tuple

from huggingface_hub import hf_hub_download

autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda").requires_grad_(False)
#%%
# # Download a specific file from dataset_aim_expert
# file_path = hf_hub_download(
#     repo_id="TeaPearce/CounterStrike_Deathmatch",
#     filename="dataset_aim_expert.tar",
#     repo_type="dataset"
# )

#%%
# Load with h5py
filename = "dataset_aim_expert/hdf5_aim_july2021_expert_1.hdf5"
def read_frames_and_actions(filename):
    with h5py.File(filename, "r") as file:
        # Extract frames and actions as NumPy arrays
        frames, actions = [], []
        for i in range(len(file)//4):
            frame = cv2.resize(file[f"frame_{i}_x"], (256, 256), interpolation=cv2.INTER_AREA)

            frames.append(frame)
            actions.append(file[f"frame_{i}_xaux"][:])
    
        # Stack frames and actions along a new axis (e.g., axis=0)
        frames = np.stack(frames, axis=0)
        actions = np.stack(actions, axis=0)

    return frames, actions

@torch.no_grad()
def encode_frames(autoencoder, frames, actions):
    # TODO: add caching to the autoencoder to make sure it can work with long sequences
    stack_size = 32*6
    n_stacks = len(frames) // stack_size
    frames = frames[:n_stacks * stack_size] # for simplicity we are going to ignore the last set of frames
    actions = actions[:n_stacks * stack_size]

    actions = einops.rearrange(actions, '(b t m) a-> b t m a', b=n_stacks, m=6).mean(-2)


    frames = einops.rearrange(frames, '(b t) h w c -> b c t h w', b=n_stacks)
    frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
    frames = torch.tensor(frames).float().to("cuda")

    autoencoder.enable_slicing()
    encoded_frames = autoencoder.encode(frames).latent_dist
    means, logvars = encoded_frames.mean.cpu().numpy(), encoded_frames.logvar.cpu().numpy()

    for mean, logvar in zip(means, logvars):
        yield {"mean": mean, "logvar": logvar, "actions": actions}
