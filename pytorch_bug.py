#%%
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE
import einops
import warnings


class DiagonalDiffusionMask:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, b, h, q_idx, kv_idx):
        q_idx, kv_idx = q_idx // self.image_size, kv_idx // self.image_size
        return q_idx >= kv_idx

def make_infer_mask(batch_size, num_heads, n_frames, image_size):
    # image size is the number of pixels (height x width)

    diagonal_diffusion_mask = DiagonalDiffusionMask(image_size)

    num_blocks_in_row = torch.arange(1, n_frames+1, dtype=torch.int32, device="cuda")
    num_blocks_in_row = einops.repeat(num_blocks_in_row, '... -> b h ...', b=batch_size, h=num_heads)

    causal_mask = torch.tril(torch.ones(n_frames, n_frames))
    col_indices = torch.arange(n_frames).expand(n_frames,n_frames) * causal_mask
    col_indices = einops.repeat(col_indices, '... -> b h ...', b=batch_size, h=num_heads).cuda().to(torch.int32)

    return BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=image_size)
    
    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 1
num_heads = 2
n_frames = 4
head_dim = 8


def run_test(image_size):
    sequence_length =  n_frames * image_size

    block_mask = make_infer_mask(batch_size, num_heads, n_frames, image_size)

    q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
    
    if block_mask is not None:
        flex_attention(q, k, v, block_mask=block_mask)
    print(f"Test passed for image_size: {image_size}")


# First run with image_size = 128
run_test(128)
run_test(16)

# Second run with image_size = 256 (this should reproduce the crash)
run_test(256)
# %%
