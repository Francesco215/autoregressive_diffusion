#%%
import torch
import time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE
from matplotlib import pyplot as plt
import gc

from edm2.attention.attention_masking import TrainingMask, make_train_mask

device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Your code (with modifications for sequence_length)
batch_size = 2
num_heads = 8
n_frames = 4
image_size = 128
head_dim = 16
sequence_length = 2 * n_frames * image_size # Removed multiplication by image_size

# q = Tensor[batch_size, num_heads, sequence_length, head_dim]
# però sequence_length = 2 x number_frames x image_size


# define the two functions
block_mask = make_train_mask(batch_size, num_heads, n_frames, image_size)
mask_function = TrainingMask(n_frames, image_size)
block_mask_old = create_block_mask(mask_function, B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length)


# ------ TESTING AND BENCHMARKING

# step1: check by eye the block mask
plt.imshow(block_mask_old.to_dense()[0,0].cpu())
plt.show()
plt.imshow(block_mask.to_dense()[0,0].cpu())
plt.show()
#%%

#%% 

# step2: check that the two block masks lead to the same results
q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*1
k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*1
v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*1
out1=flex_attention(q,k,v, block_mask=block_mask) 
out2=flex_attention(q,k,v, block_mask=block_mask_old)
out = out1-out2 # it's equal to zero. it works.
print((out==0).all()) #True
#%%

# manual_masking = torch.zeros(size=(batch_size, num_heads, sequence_length, sequence_length), dtype=torch.bool, device='cuda')
# for i in range(sequence_length):
#     for j in range(sequence_length):
#         manual_masking[:,:,i,j]=mask_function(0,0,i,j)

# plt.imshow(manual_masking.to_dense()[0,0].cpu())
        

repeated_mask = block_mask_old.to_dense()[0,0].bool().repeat_interleave(_DEFAULT_SPARSE_BLOCK_SIZE,0).repeat_interleave(_DEFAULT_SPARSE_BLOCK_SIZE,1)
plt.imshow(repeated_mask.cpu())
plt.show()



out3 = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=repeated_mask)
out = out1-out3 # it's equal to zero. it works.
print((out==0).all()) #should be true, but it's false. why??



#%%
inference_mask=make_AR_BlockMask_Inference(batch_size, num_heads, n_frames, image_size)
plt.imshow(inference_mask.to_dense()[0,0].cpu())

#%%
# step3: see if it compiles
#%%
import torch
import time
import einops
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE
from matplotlib import pyplot as plt
import numpy as np
import gc

batch_size = 2
num_heads = 8
n_frames = 8
image_size = 64  # this is the widthxheight
head_dim = 16
sequence_length = n_frames * image_size # Removed multiplication by image_size

device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def diagonal_diffusion_mask(b, h, q_idx, kv_idx):
    q_idx, kv_idx = q_idx // image_size, kv_idx // image_size
    
    return q_idx >= kv_idx

ar_mask=create_block_mask(diagonal_diffusion_mask, B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length) 
q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*10
k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*10
v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*10
# @torch.compile
def autoregressive_diffusion_attention(q, k, v):
    return flex_attention(q,k,v, block_mask=ar_mask) 

autoregressive_diffusion_attention(q,k,v)
#%%
del block_mask_old
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
#%%

# step4: benchmark with respect to Standard Attention
@torch.compile
def standard_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
#%%
# Benchmarking function
def benchmark(attn_func, num_iterations=100):
    # Warmup
    for _ in range(10):
        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        attn_func(q, k, v).mean().item()
    torch.cuda.synchronize()

    # Memory profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        attn_func(q, k, v).mean().item()
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_per_iteration = elapsed_time / num_iterations
    # Memory profiling
    if torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # in MB
        max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)  # in MB
    else:
        max_memory_allocated = 0
        max_memory_reserved = 0

    return avg_time_per_iteration, max_memory_allocated, max_memory_reserved

#%%

# Benchmark AR diffusion attention
ar_time, ar_memory_allocated, ar_memory_reserved = benchmark(autoregressive_diffusion_attention)
print(f"Karras FlexAttention: {ar_time:.6f} seconds per iteration")
print(f"Karras FlexAttention: Max Memory Allocated: {ar_memory_allocated:.2f} MB, Max Memory Reserved: {ar_memory_reserved:.2f} MB")

# Benchmark Standard Attention
standard_time, standard_memory_allocated, standard_memory_reserved = benchmark(standard_attention)
print(f"Standard Attention: {standard_time:.6f} seconds per iteration")
print(f"Standard Attention: Max Memory Allocated: {standard_memory_allocated:.2f} MB, Max Memory Reserved: {standard_memory_reserved:.2f} MB")

# %%