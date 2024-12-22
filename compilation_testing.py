#%%
import torch
import time
import einops
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE
from matplotlib import pyplot as plt
import numpy as np
import gc

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Your code (with modifications for sequence_length)
batch_size = 2
num_heads = 8
n_frames = 32
image_size = 128  # this is the widthxheight
head_dim = 16
sequence_length = 2 * n_frames * image_size 

# q = Tensor[batch_size, num_heads, sequence_length, head_dim]
# però sequence_length = 2 x number_frames x image_size
def autoregressive_diffusion_mask(b, h, q_idx, kv_idx):
    q_idx, kv_idx = q_idx // image_size, kv_idx // image_size

    causal_mask_clean = q_idx >= kv_idx
    causal_mask_noisy = q_idx - n_frames > kv_idx
    domain_attention_towards_clean = kv_idx < n_frames
    mask_towards_clean = (causal_mask_clean ^ causal_mask_noisy ^ (q_idx < n_frames)) & domain_attention_towards_clean

    self_mask_noisy = (kv_idx >= n_frames) & (q_idx == kv_idx)
    return mask_towards_clean ^ self_mask_noisy ^ domain_attention_towards_clean

def make_ARBlockMask(n_frames, image_size, mask_mod=autoregressive_diffusion_mask):
    block_size = image_size 
    num_blocks_in_row = torch.arange(1, n_frames + 1, dtype=torch.int32, device=device).repeat(2)
    num_blocks_in_row = einops.repeat(num_blocks_in_row, '... -> b h ...', b=batch_size, h=num_heads)

    causal_mask = torch.tril(torch.ones(n_frames, n_frames))
    col_indices1 = torch.arange(n_frames).expand(n_frames, n_frames) * causal_mask
    col_indices2 = torch.arange(n_frames).expand(n_frames, n_frames) * causal_mask
    col_indices2 = col_indices2 + torch.diag(torch.ones(n_frames)) * n_frames

    col_indices = torch.cat((col_indices1, col_indices2))
    col_indices = torch.cat((col_indices, torch.zeros_like(col_indices)), dim=1)
    col_indices = einops.repeat(col_indices, '... -> b h ...', b=batch_size, h=num_heads).cuda().to(torch.int32)

    return BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=block_size, mask_mod=mask_mod)

# define the two functions
block_mask = make_ARBlockMask(n_frames, image_size)
block_mask_old = create_block_mask(autoregressive_diffusion_mask, B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length)


# ------ TESTING AND BENCHMARKING

# step1: check by eye the block mask
plt.imshow(block_mask_old.to_dense()[0, 0].cpu())
plt.show()
plt.imshow(block_mask.to_dense()[0, 0].cpu())
plt.show()


#%%
# step2: check that the two block masks lead to the same results
q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16) * 10
k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16) * 10
v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16) * 10
out1 = flex_attention(q, k, v, block_mask=block_mask)
out2 = flex_attention(q, k, v, block_mask=block_mask_old)
out = out1 - out2  # it's equal to zero. it works.
print((out == 0).all())  # True


#%%
# step3: see if it compiles
@torch.compile
def autoregressive_diffusion_attention(q, k, v):
    return flex_attention(q, k, v, block_mask=block_mask)

def uncompiled_autoregressive_diffusion_attention(q, k, v):
    return flex_attention(q,k,v, block_mask=block_mask)

autoregressive_diffusion_attention(q, k, v)
uncompiled_autoregressive_diffusion_attention(q,k,v)

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

def uncompiled_standard_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

#%%
# Benchmarking function
def benchmark(attn_func, num_iterations=100):
    # Warmup
    for _ in range(10):
        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
        attn_func(q, k, v)
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
        attn_func(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_per_iteration = elapsed_time / num_iterations
    # Memory profiling
    if torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
        max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # in MB
    else:
        max_memory_allocated = 0
        max_memory_reserved = 0

    return avg_time_per_iteration, max_memory_allocated, max_memory_reserved

#%%
# COMPILED
# Benchmark AR diffusion attention
ar_time, ar_memory_allocated, ar_memory_reserved = benchmark(autoregressive_diffusion_attention)
print(f"Compiled Karras FlexAttention: {ar_time:.6f} seconds per iteration")
print(f"Compiled Karras FlexAttention: Max Memory Allocated: {ar_memory_allocated:.2f} MB, Max Memory Reserved: {ar_memory_reserved:.2f} MB")

# Benchmark Standard Attention
standard_time, standard_memory_allocated, standard_memory_reserved = benchmark(standard_attention)
print(f"Compiled Standard Attention: {standard_time:.6f} seconds per iteration")
print(f"Compiled Standard Attention: Max Memory Allocated: {standard_memory_allocated:.2f} MB, Max Memory Reserved: {standard_memory_reserved:.2f} MB")

# UNCOMPILED
# Benchmark AR diffusion attention
ar_time, ar_memory_allocated, ar_memory_reserved = benchmark(uncompiled_autoregressive_diffusion_attention)
print(f"Uncompiled Karras FlexAttention: {ar_time:.6f} seconds per iteration")
print(f"Uncompiled Karras FlexAttention: Max Memory Allocated: {ar_memory_allocated:.2f} MB, Max Memory Reserved: {ar_memory_reserved:.2f} MB")

# Benchmark Standard Attention
standard_time, standard_memory_allocated, standard_memory_reserved = benchmark(uncompiled_standard_attention)
print(f"Uncompiled Standard Attention: {standard_time:.6f} seconds per iteration")
print(f"Uncompiled Standard Attention: Max Memory Allocated: {standard_memory_allocated:.2f} MB, Max Memory Reserved: {standard_memory_reserved:.2f} MB")

# %%
