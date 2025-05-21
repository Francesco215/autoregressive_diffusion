#%%
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _DEFAULT_SPARSE_BLOCK_SIZE
import einops
from edm2.attention.attention_masking import make_train_mask

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 2
num_heads = 8
n_frames = 4
image_size = 128
head_dim = 16
sequence_length = 2 * n_frames * image_size

# --- Generate block masks ---
block_mask = make_train_mask(batch_size, num_heads, n_frames, image_size)

# --- Create random inputs ---
q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)

# --- Apply flex attention ---
out1 = flex_attention(q, k, v, block_mask=block_mask)

# --- Apply reshape the random imputs
q = einops.rearrange(q, 'b m (t hw) c -> (b t) m hw c', hw = image_size)
k = einops.rearrange(k, 'b m (t hw) c -> (b t) m hw c', hw = image_size)
v = einops.rearrange(v, 'b m (t hw) c -> (b t) m hw c', hw = image_size)

# --- Apply flash attention ---
out2 = F.scaled_dot_product_attention(q,k,v)


out2 = einops.rearrange(out2, '(b s t) m hw c -> (s b) t m hw c', s = 2, b = batch_size)
out1 = einops.rearrange(out1, 'b m (s t hw) c -> (s b) t m hw c', s = 2, hw = image_size)
diff_std = (out2 - out1).std(dim=(0, 2, 3, 4))
print(diff_std[0]) # it retuns 1e-5!

# %%
