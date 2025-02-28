#%%
# Import necessary libraries
import einops
import numpy as np
import torch
from diffusers import AutoencoderKLMochi
import os
import h5py
import cv2
import tarfile
from streaming import MDSWriter
from tqdm import tqdm
import threading
from huggingface_hub import hf_hub_download, HfApi
import re

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
def encode_frames(autoencoder, frames, actions, stack_size):
    # TODO: add caching to the autoencoder source code to make sure it can work with long sequences
    n_stacks = len(frames) // stack_size
    frames = frames[:n_stacks * stack_size] # for simplicity we are going to ignore the last set of frames
    actions = actions[:n_stacks * stack_size]

    actions = einops.rearrange(actions, '(b t m) a-> b t m a', b=n_stacks, m=6).mean(-2)


    frames = einops.rearrange(frames, '(b t) h w c -> b c t h w', b=n_stacks)
    frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
    frames = torch.tensor(frames).to(torch.float16).to("cuda")

    autoencoder.enable_slicing()
    encoded_frames = autoencoder.encode(frames).latent_dist
    means, logvars = encoded_frames.mean.cpu().numpy(), encoded_frames.logvar.cpu().numpy()

    for mean, logvar, action in zip(means, logvars, actions):
        yield {"mean": mean, "logvar": logvar, "action": action}


def download_tar_file(hf_repo_id, hf_filename):
    tar_file_path = hf_hub_download(hf_repo_id, hf_filename, repo_type = "dataset")

    # Extract the downloaded .tar file
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(f"/tmp/{hf_filename.split('.')[0]}")  

    #delete the tar file
    os.remove(tar_file_path)

def compress_huggingface_filename(save_folder, stack_size):
    file_list = os.listdir(save_folder)

    for file in file_list:
        frames, actions = read_frames_and_actions(f"{save_folder}/{file}")
        os.remove(f"{save_folder}/{file}")

        yield from encode_frames(autoencoder, frames, actions, stack_size)


def write_mds(save_folder, mds_dirname, stack_size):
    columns = {'mean': 'ndarray', 'logvar': 'ndarray', 'action': 'ndarray'}

    n_clips, n_frames = len(os.listdir(save_folder)), 1000
    total = (n_clips * n_frames) // stack_size

    # the mdswriter uploads the data to the s3 bucket and deletes the local files
    with MDSWriter(out=mds_dirname, columns=columns, compression='zstd') as writer:
        for encoded_frame in tqdm(compress_huggingface_filename(save_folder, stack_size), total=total):
            writer.write(encoded_frame)
    
    os.rmdir(save_folder)

#%%
api = HfApi()

hf_repo_id="TeaPearce/CounterStrike_Deathmatch"
dataset_filenames = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")

#have to filter out some of the data because its's saved slightly differently...
hf_filenames = [f for f in dataset_filenames if re.match(r"^hdf5_dm_july2021_.*_to_.*\.tar$", f)]
stack_size = 64*6

autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float16).to("cuda").requires_grad_(False)

#%%
# Download the first tar file
download_tar_file(hf_repo_id, hf_filenames[0])

for i in range(len(hf_filenames)):
    save_folder = f"/tmp/{hf_filenames[i].split('.')[0]}"
    
    # Start downloading the next tar file (if there is one)
    if i < len(hf_filenames) - 1:
        next_tar_file = hf_filenames[i+1]
        download_thread = threading.Thread(
            target=download_tar_file,
            args=(hf_repo_id, next_tar_file)
        )
        download_thread.start()
    
    # Process the current tar file
    write_mds(save_folder, f"s3://counter-strike-data/dataset_small/{hf_filenames[i].split('.')[0]}", stack_size)
    
    # Wait for the next download to finish (if applicable)
    if i < len(hf_filenames) - 1:
        download_thread.join()