#%%
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask
from edm2.attention.attention_masking import TrainingMask, make_train_mask, make_infer_mask
from edm2.attention.attention_modules import compiled_flex_attention

image_width, n_frames = 4, 64
batch_size = 4
num_heads = 4
image_size= image_width **2
channels = 16

y = torch.randn(3, batch_size, n_frames, num_heads, image_size, channels)

y[:,:,-4]*=100000

y=einops.rearrange(y, 's b t m n c -> s b m (t n) c').to("cuda")
q, k, v = y.unbind(0)

infer_score, infer_mask = make_infer_mask(batch_size, num_heads, n_frames, image_size)
out_i = compiled_flex_attention(q,k,v, infer_score, infer_mask)
out_i = einops.rearrange(out_i, 'b m (t n) c -> b t m n c', t=n_frames)
print(out_i.mean(dim=(0,2,3,4)))
# out_i = compiled_flex_attention(q,k,v, infer_score, infer_mask)

# %%

y = torch.randn(3, batch_size, n_frames*2, num_heads, image_size, channels, device = "cuda")
y[:,:,5]*=100000
y=einops.rearrange(y, 's b t m n c -> s b m (t n) c').to("cuda")
q, k, v = y.unbind(0)

train_score, train_mask = make_train_mask(batch_size, num_heads, n_frames, image_size)
out_t = compiled_flex_attention(q,k,v, train_score, train_mask)
out_t = einops.rearrange(out_t, 'b m (t n) c -> b t m n c', t=n_frames*2)
print(out_t.std(dim=(0,2,3,4))>1e2)
# %%
y = torch.randn(3, batch_size, n_frames, num_heads, image_size, channels)
y[:,:,-1]/=0
y=einops.rearrange(y, 's b t m n c -> s b m (t n) c').to("cuda")
q, k, v = y.unbind(0)




#%%
from datasets import load_dataset
import h5py
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2 
# Load the dataset
dere = load_dataset('DereWah/mk64-steering')

# Get the HDF5 data as bytes from the first entry
hdf5_bytes = dere['train'][0]['hdf5']

# Use io.BytesIO to treat the bytes as a file
hdf5_file = io.BytesIO(hdf5_bytes)

# Open the in-memory file with h5py
with h5py.File(hdf5_file, 'r') as f:
    # --- To find the correct key for the pixel data ---
    # You can list all the keys in the HDF5 file like this:
    keys = list(f.keys())
    print("Keys in the HDF5 file:", keys)
    pixel_data=f[keys[0]]
    frame = cv2.resize(pixel_data[:], (256, 256), interpolation=cv2.INTER_AREA)



    # The pixel_data is now a NumPy array
    print("Shape of the pixel data:", frame.shape)
    print("Data type of pixel data:", pixel_data.dtype)

#%%
# --- Optional: Display the first frame to verify ---
if len(pixel_data.shape) == 4: # Assuming shape is (frames, height, width, channels)
    plt.imshow(pixel_data[0])
    plt.title("First Frame")
    plt.axis('off') # Hide axes
    plt.show()# %%
