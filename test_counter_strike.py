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
import tarfile
from streaming import MDSWriter
from tqdm import tqdm
from typing import Tuple

from huggingface_hub import hf_hub_download

autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda").requires_grad_(False)

repo_id="TeaPearce/CounterStrike_Deathmatch"
filename = "dataset_aim_expert.tar"
#%%
# Load with h5py
def read_frames_and_actions(filename):
    with h5py.File(filename, "r") as file:
        # Extract frames and actions as NumPy arrays
        frames, actions = [], []
        for i in range(len(file)//4):
            frame = cv2.resize(file[f"frame_{i}_x"][:], (256, 256), interpolation=cv2.INTER_AREA)

            frames.append(frame)
            actions.append(file[f"frame_{i}_xaux"][:])
    
        # Stack frames and actions along a new axis (e.g., axis=0)
        frames = np.stack(frames, axis=0)
        actions = np.stack(actions, axis=0)

    return frames, actions

@torch.no_grad()
def encode_frames(autoencoder, frames, actions):
    # TODO: add caching to the autoencoder source code to make sure it can work with long sequences
    stack_size = 32*6
    n_stacks = len(frames) // stack_size
    frames = frames[:n_stacks * stack_size] # for simplicity we are going to ignore the last set of frames
    actions = actions[:n_stacks * stack_size]

    actions = einops.rearrange(actions, '(b t m) a-> b t m a', b=n_stacks, m=6).mean(-2)


    frames = einops.rearrange(frames, '(b t) h w c -> b c t h w', b=n_stacks)
    frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
    frames = torch.tensor(frames).float().to("cuda")

    autoencoder.enable_slicing()
    # print(frames.shape)
    encoded_frames = autoencoder.encode(frames).latent_dist
    means, logvars = encoded_frames.mean.cpu().numpy(), encoded_frames.logvar.cpu().numpy()

    for mean, logvar, action in zip(means, logvars, actions):
        yield {"mean": mean, "logvar": logvar, "action": action}

def compress_huggingface_filename(repo_id, filename):
    tar_file_path = hf_hub_download(repo_id, filename, repo_type = "dataset")

    # Extract the downloaded .tar file
    save_folder = f"/tmp/{filename.split('.')[0]}"
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall("/tmp/")  

    #delete the tar file
    os.remove(tar_file_path)

    # get the path of each file
    file_list = os.listdir(save_folder)

    for file in file_list:
        frames, actions = read_frames_and_actions(f"{save_folder}/{file}")
        os.remove(f"{save_folder}/{file}")

        yield from encode_frames(autoencoder, frames, actions)


def write_mds(hf_repo_id, hf_filename, mds_dirname):
    columns = {'mean': 'ndarray', 'logvar': 'ndarray', 'action': 'ndarray'}
    os.makedirs(mds_dirname, exist_ok=True)
    with MDSWriter(out=mds_dirname, columns=columns, compression='zstd') as writer:
        for encoded_frame in tqdm(compress_huggingface_filename(hf_repo_id, hf_filename)):
            writer.write(encoded_frame)

#%%
write_mds(repo_id, filename, "counter_strike")
# %%
