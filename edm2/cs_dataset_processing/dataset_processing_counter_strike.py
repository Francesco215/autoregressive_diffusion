#%%
# Import necessary libraries
import einops
import numpy as np
import torch
import os
import h5py
import cv2
import tarfile
from streaming import MDSWriter
from tqdm import tqdm
import threading
from huggingface_hub import hf_hub_download, HfApi
import re
import tempfile
from edm2.vae import VAE
from diffusers import AutoencoderKL

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

# @torch.no_grad()
# def encode_frames(autoencoder, frames, actions):
#     # TODO: add caching to the autoencoder source code to make sure it can work with long sequences

#     frames = einops.rearrange(frames, 't h w c -> t c h w')
#     frames = torch.tensor(frames).to("cuda").to(torch.float32)
#     frames = frames / 255  # Normalize from (0,255) to (0,1)
#     mean, logvar = [], []
#     for batch in frames.split(64):
#         out = autoencoder.encode(batch)['latent_dist'] 
#         mean.append(out.mean), logvar.append(out.logvar)

#     mean, logvar = torch.cat(mean,dim=0), torch.cat(logvar,dim=0)
#     mean, logvar = mean.to(torch.float16), logvar.to(torch.float16)
#     out_dict = {'mean':mean.cpu().numpy(), 'logvar':logvar.cpu().numpy(), 'action':actions}
#     return out_dict

@torch.no_grad()
def encode_frames(autoencoder, frames, actions):
    # TODO: add caching to the autoencoder source code to make sure it can work with long sequences

    frames = einops.rearrange(frames, 't h w c -> c t h w')
    frames = torch.tensor(frames).to("cuda")
    frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
    frames = autoencoder.encode_long_sequence(frames.unsqueeze(0), split_size=512)
    frames = frames.to(torch.float16)
    out_dict = {'mean':frames[0].cpu().numpy(), 'action':actions}
    return out_dict

def download_tar_file(hf_repo_id, hf_filename):
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        # Download the tar file to the temporary cache directory
        tar_file_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type="dataset",
            cache_dir=temp_cache_dir
        )
        # Extract the tar file to /tmp/{filename}
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(f"/tmp/{hf_filename.split('.')[0]}")


def compress_huggingface_filename(save_folder):
    file_list = os.listdir(save_folder)

    for file in file_list:
        frames, actions = read_frames_and_actions(f"{save_folder}/{file}")
        os.remove(f"{save_folder}/{file}")

        yield encode_frames(vae, frames, actions)


def write_mds(save_folder, mds_dirname):
    columns = {'mean': 'ndarray', 'action': 'ndarray'}

    n_clips = len(os.listdir(save_folder))

    # the mdswriter uploads the data to the s3 bucket and deletes the local files
    with MDSWriter(out=mds_dirname, columns=columns, compression='zstd') as writer:
        for encoded_frame in tqdm(compress_huggingface_filename(save_folder), total=n_clips):
            writer.write(encoded_frame)
    
    os.rmdir(save_folder)

#%%
api = HfApi()

hf_repo_id="TeaPearce/CounterStrike_Deathmatch"
dataset_filenames = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")

#have to filter out some of the data because its's saved slightly differently...
hf_filenames = [f for f in dataset_filenames if re.match(r"^hdf5_dm_july2021_.*_to_.*\.tar$", f)][10:]

vae = VAE.from_pretrained("s3://autoregressive-diffusion/saved_models/vae_cs_102354.pt").to("cuda")
# vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to("cuda")


#%%
# Download the first tar file
# download_tar_file(hf_repo_id, hf_filenames[0])

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
    write_mds(save_folder, f"s3://counter-strike-data/vae_40M/{hf_filenames[i].split('.')[0]}")
    
    # Wait for the next download to finish (if applicable)
    if i < len(hf_filenames) - 1:
        download_thread.join()
print("cinokete")
# %%
