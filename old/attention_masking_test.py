#%%
import torch
from torch.nn import functional as F
import time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE
from matplotlib import pyplot as plt
import gc
import einops

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

from functools import lru_cache
import einops
import warnings
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE


class TrainingMask:
    def __init__(self, n_frames, image_size):
        # image_size is the number of pixels (height x width)
        self.n_frames = torch.tensor(n_frames, dtype = torch.int32, device="cuda")
        self.image_size = torch.tensor(image_size, dtype = torch.int32, device = "cuda")


    def __call__(self, b, h, q_idx, kv_idx):
        q_idx, kv_idx = q_idx // self.image_size, kv_idx // self.image_size
        
        causal_mask_clean = q_idx >= kv_idx
        causal_mask_noisy = q_idx - self.n_frames > kv_idx 
        domain_attention_towards_clean = kv_idx < self.n_frames
        mask_towards_clean = (causal_mask_clean ^ causal_mask_noisy ^ (q_idx < self.n_frames)) & domain_attention_towards_clean

        self_mask_noisy = (kv_idx >= self.n_frames) & (q_idx == kv_idx)
        return mask_towards_clean ^ self_mask_noisy ^ domain_attention_towards_clean


@lru_cache(maxsize=16)
def make_train_mask(batch_size, num_heads, n_frames, image_size):
    # image_size is the number of pixels (height x width)
    training_mask = TrainingMask(n_frames, image_size)

    if image_size<_DEFAULT_SPARSE_BLOCK_SIZE:
        if n_frames*image_size%_DEFAULT_SPARSE_BLOCK_SIZE!=0:
            warnings.warn("The image size must be a divisor of the default block size ({_DEFAULT_SPARSE_BLOCK_SIZE}), got image_size:{image_size} and n_frames:{n_frames}\n returning None")
            return None

        n_frames = n_frames*image_size//_DEFAULT_SPARSE_BLOCK_SIZE
        image_size = _DEFAULT_SPARSE_BLOCK_SIZE
    
    num_blocks_in_row = torch.arange(1, n_frames+1, dtype=torch.int32, device="cuda").repeat(2)
    num_blocks_in_row = einops.repeat(num_blocks_in_row, '... -> b h ...', b=batch_size, h=num_heads)

    causal_mask = torch.tril(torch.ones(n_frames, n_frames))
    col_indices1 = torch.arange(n_frames).expand(n_frames,n_frames) * causal_mask
    col_indices2 = torch.arange(n_frames).expand(n_frames,n_frames) * causal_mask
    col_indices2 = col_indices2 + torch.diag(torch.ones(n_frames)) * n_frames

    col_indices = torch.cat((col_indices1, col_indices2))
    col_indices = torch.cat((col_indices,torch.zeros_like(col_indices)), dim=1)
    col_indices = einops.repeat(col_indices, '... -> b h ...', b=batch_size, h=num_heads).cuda().to(torch.int32)

    # return BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=image_size)
    return BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=image_size, mask_mod=training_mask)



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

# step2: check that the two block masks lead to the same results
q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*1
k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*1
v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)*1
out1=flex_attention(q,k,v, block_mask=block_mask) 
out2=flex_attention(q,k,v, block_mask=block_mask_old)
print((out1-out2).std()) # tensor(0., device='cuda:0', dtype=torch.float16)
#%%

# manual_masking = torch.zeros(size=(batch_size, num_heads, sequence_length, sequence_length), dtype=torch.bool, device='cuda')
# for i in range(sequence_length):
#     for j in range(sequence_length):
#         manual_masking[:,:,i,j]=mask_function(0,0,i,j)

# plt.imshow(manual_masking.to_dense()[0,0].cpu())
        

repeated_mask = block_mask_old.to_dense()[0,0].bool().repeat_interleave(_DEFAULT_SPARSE_BLOCK_SIZE,0).repeat_interleave(_DEFAULT_SPARSE_BLOCK_SIZE,1)
plt.imshow(repeated_mask.cpu())
plt.show()



out3 = F.scaled_dot_product_attention(q,k,v,attn_mask=repeated_mask)
out = print((out1-out3).std()) # tensor(6.3181e-05, device='cuda:0', dtype=torch.float16)

#%%
class DiagonalMask:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, b, h, q_idx, kv_idx):
        q_idx, kv_idx = q_idx // self.image_size, kv_idx // self.image_size
        return q_idx == kv_idx

q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)

block_mask_diag = create_block_mask(DiagonalMask(image_size), B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length)
out4 = flex_attention(q,k,v, block_mask=block_mask_diag) 

q = einops.rearrange(q, 'b m (t hw) c -> (b t) m hw c', hw = image_size)
k = einops.rearrange(k, 'b m (t hw) c -> (b t) m hw c', hw = image_size)
v = einops.rearrange(v, 'b m (t hw) c -> (b t) m hw c', hw = image_size)
out4 = einops.rearrange(out4, 'b m (t hw) c -> (b t) m hw c', hw = image_size)

out5 = F.scaled_dot_product_attention(q,k,v)

out=print((out4-out5).std())


plt.imshow(block_mask_diag.to_dense()[0,0].cpu())

#%%

out4 = einops.rearrange(out4, '(b t) m hw c -> b t m hw c', b = batch_size)
out1 = einops.rearrange(out1, 'b m (t hw) c -> b t m hw c', hw = image_size)
#%%
print((out4-out1).std(dim=(0,1,2,3))) # tensor([0.1815, 0.1780, 0.1809, 0.1757, 0.1819, 0.1748, 0.1783, 0.1813, 0.1895, 0.1749, 0.1774, 0.1741, 0.1842, 0.1805, 0.1879, 0.1758],device='cuda:0', dtype=torch.float16)


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