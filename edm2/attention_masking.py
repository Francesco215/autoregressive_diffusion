import einops
import warnings
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE


class AutoregressiveDiffusionMask:

    def __init__(self, n_frames, image_size):
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


def make_train_mask(batch_size, num_heads, n_frames, image_size):
    # image_size is the number of pixels (height x width)
    autoregressive_diffusion_mask = AutoregressiveDiffusionMask(n_frames, image_size)

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

    return BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=image_size, mask_mod=autoregressive_diffusion_mask)


class DiagonalDiffusionMask:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, b, h, q_idx, kv_idx):
        q_idx, kv_idx = q_idx // self.image_size, kv_idx // self.image_size
        return q_idx >= kv_idx

def make_infer_mask(batch_size, num_heads, n_frames, image_size):
    # image size is the number of pixels (height x width)

    diagonal_diffusion_mask = DiagonalDiffusionMask(image_size)

    if n_frames*image_size<_DEFAULT_SPARSE_BLOCK_SIZE:
        warnings.warn(f"The masking matrix must be at least the size of the default block size,\ngot {n_frames*image_size} and the default block size is {_DEFAULT_SPARSE_BLOCK_SIZE}\n returning None")
        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(diagonal_diffusion_mask(b, h, q_idx, kv_idx), score, torch.full_like(score, -float('inf')))
        return score_mod, None
    if image_size<_DEFAULT_SPARSE_BLOCK_SIZE:
        if n_frames*image_size%_DEFAULT_SPARSE_BLOCK_SIZE!=0:
            sequence_length = n_frames*image_size
            warnings.warn(f"\nThe image size must be a divisor of the default block size ({_DEFAULT_SPARSE_BLOCK_SIZE})\ngot image_size:{image_size} and n_frames:{n_frames}\n using {(sequence_length**2 * batch_size * num_heads)/1e6}M of memory")
            return None, create_block_mask(diagonal_diffusion_mask, B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length)
        n_frames = n_frames*image_size//_DEFAULT_SPARSE_BLOCK_SIZE
        image_size = _DEFAULT_SPARSE_BLOCK_SIZE 

    num_blocks_in_row = torch.arange(1, n_frames+1, dtype=torch.int32, device="cuda")
    num_blocks_in_row = einops.repeat(num_blocks_in_row, '... -> b h ...', b=batch_size, h=num_heads)

    causal_mask = torch.tril(torch.ones(n_frames, n_frames))
    col_indices = torch.arange(n_frames).expand(n_frames,n_frames) * causal_mask
    col_indices = einops.repeat(col_indices, '... -> b h ...', b=batch_size, h=num_heads).cuda().to(torch.int32)

    return None, BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=image_size, mask_mod=diagonal_diffusion_mask)
    
    