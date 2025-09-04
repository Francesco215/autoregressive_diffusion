#%%
# Import necessary libraries
import einops
import numpy as np
import os
import h5py
import cv2
from streaming import MDSWriter
from tqdm import tqdm
import threading
import re
import tempfile
from edm2.vae import VAE
import io
from datasets import load_dataset

# Load with h5py
def read_frames_and_actions(hdf5):
    hdf5_file = io.BytesIO(hdf5)
    with h5py.File(hdf5_file, "r") as file:
        # Extract frames and actions as NumPy arrays
        frames, actions = [], []
        for i in range(len(file)//2):
            frames.append( file[f"frame_{i}_x"][:])
            actions.append(file[f"frame_{i}_y"][:])
    
        # Stack frames and actions along a new axis (e.g., axis=0)
        frames = np.stack(frames, axis=0)
        actions = np.stack(actions, axis=0)

    return frames, actions


def dataset_iterator(train):
    for file in train:
        frames, actions = read_frames_and_actions(file['hdf5'])

        yield {"frames": frames, "actions": actions}


def write_mds(dataset, mds_dirname):
    columns = {'frames': 'ndarray', 'actions': 'ndarray'}

    n_clips = len(dataset)

    # the mdswriter uploads the data to the s3 bucket and deletes the local files
    with MDSWriter(out=mds_dirname, columns=columns, compression='zstd') as writer:
        for encoded_frame in tqdm(dataset_iterator(dataset), total=n_clips):
            writer.write(encoded_frame)
    

#%%
# Download the first tar file
# download_tar_file(hf_repo_id, hf_filenames[0])


dere = load_dataset('DereWah/mk64-steering')
write_mds(dere['train'], f"s3://mario-kart-data/raw")
    
# %%
