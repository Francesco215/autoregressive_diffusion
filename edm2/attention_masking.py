import torch
import einops
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def make_AR_BlockMask(batch_size, num_heads, n_frames, image_size, mask_mod=None):
    # image size is the number of pixels (height x width)

    def autoregressive_diffusion_mask(b, h, q_idx, kv_idx):
        q_idx, kv_idx = q_idx // image_size, kv_idx // image_size
        
        causal_mask_clean = q_idx >= kv_idx
        causal_mask_noisy = q_idx - n_frames > kv_idx 
        domain_attention_towards_clean = kv_idx < n_frames
        mask_towards_clean = (causal_mask_clean ^ causal_mask_noisy ^ (q_idx < n_frames)) & domain_attention_towards_clean

        self_mask_noisy = (kv_idx >= n_frames) & (q_idx == kv_idx)
        return mask_towards_clean ^ self_mask_noisy ^ domain_attention_towards_clean

    if mask_mod is None: mask_mod = autoregressive_diffusion_mask

    if image_size<_DEFAULT_SPARSE_BLOCK_SIZE: #TODO: test this!
        # return make_AR_BlockMask(n_frames*image_size//_DEFAULT_SPARSE_BLOCK_SIZE, _DEFAULT_SPARSE_BLOCK_SIZE, mask_mod=autoregressive_diffusion_mask)
        if n_frames*image_size%_DEFAULT_SPARSE_BLOCK_SIZE==0:
            sequence_length = n_frames*image_size*2
            return create_block_mask(autoregressive_diffusion_mask, B=batch_size, H=num_heads, Q_LEN=sequence_length, KV_LEN=sequence_length) 
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

    return BlockMask.from_kv_blocks(num_blocks_in_row, col_indices, BLOCK_SIZE=image_size, mask_mod=mask_mod)
