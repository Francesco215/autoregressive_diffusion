#%%
import torch
import time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask 
from matplotlib import pyplot as plt
import numpy as np

device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Your code (with modifications for sequence_length)
batch_size = 4
num_heads = 8
number_frames = 4
image_size = 4  # this is the widthxheight
head_dim = 16
sequence_length = 2 * number_frames * image_size # Removed multiplication by image_size

# q = Tensor[batch_size, num_heads, sequence_length, head_dim]
# però sequence_length = 2 x number_frames x image_size
def autoregressive_diffusion_mask(b, h, q_idx, kv_idx):
    q_idx, kv_idx = q_idx // image_size, kv_idx // image_size
    
    causal_mask_clean = q_idx >= kv_idx
    causal_mask_noisy = q_idx - number_frames > kv_idx 
    domain_attention_towards_clean = kv_idx < number_frames
    mask_towards_clean = (causal_mask_clean ^ causal_mask_noisy ^ (q_idx < number_frames)) & domain_attention_towards_clean

    self_mask_noisy = (kv_idx >= number_frames) & (q_idx == kv_idx)
    return mask_towards_clean ^ self_mask_noisy ^ domain_attention_towards_clean

block_mask = create_block_mask(autoregressive_diffusion_mask, B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length)


# ------ TESTING AND BENCHMARKING

# Plotting function
a = np.zeros((sequence_length, sequence_length),dtype=np.float32)
for i in range(sequence_length):
    for j in range(sequence_length):
        a[i,j] = autoregressive_diffusion_mask(0,0,i,j)

plt.imshow(a)
#%% 
@torch.compile
def autoregressive_diffusion_attention(q, k, v):
    return flex_attention(q,k,v, block_mask=block_mask) 

# Standard Attention for comparison
@torch.compile
def standard_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
#%%
# Benchmarking function
def benchmark(attn_func, num_iterations=100):
    # Warmup
    for _ in range(10):
        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        attn_func(q, k, v)
    torch.cuda.synchronize()

    # Memory profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        attn_func(q, k, v)
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

# Benchmark Karras FlexAttention
karras_time, karras_memory_allocated, karras_memory_reserved = benchmark(autoregressive_diffusion_attention)
print(f"Karras FlexAttention: {karras_time:.6f} seconds per iteration")
print(f"Karras FlexAttention: Max Memory Allocated: {karras_memory_allocated:.2f} MB, Max Memory Reserved: {karras_memory_reserved:.2f} MB")

# Benchmark Standard Attention
standard_time, standard_memory_allocated, standard_memory_reserved = benchmark(standard_attention)
print(f"Standard Attention: {standard_time:.6f} seconds per iteration")
print(f"Standard Attention: Max Memory Allocated: {standard_memory_allocated:.2f} MB, Max Memory Reserved: {standard_memory_reserved:.2f} MB")
# %%