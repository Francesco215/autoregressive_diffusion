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
height, width = 16, 16
image_size = height*width
head_dim = 16
sequence_length = 2 * n_frames * image_size

# --- Generate block masks ---
block_mask = make_train_mask(batch_size, num_heads, n_frames, image_size)

# --- Create random inputs ---
y = torch.randn(batch_size*n_frames*2, head_dim*num_heads*3, height, width, device = device, dtype = torch.float16)
q, k, v = einops.rearrange(y, '(b t) (s m c) h w -> s b m (t h w) c', b=batch_size, s=3, m=num_heads).unbind(0)

# --- Apply flex attention ---
out1 = flex_attention(q, k, v, block_mask=block_mask)
# out1 = einops.rearrange(out1, 'b m (s t hw) c -> (s b) t m hw c', s = 2, hw = image_size)
out1 =  einops.rearrange(out1, 'b m (t h w) c -> (b t) (m c) h w', b=batch_size, h=height, w=width)
out1 = einops.rearrange(out1, '(b s t) c h w -> (s b) t c h w', b=batch_size, s = 2)

# --- Apply reshape the random imputs
q, k, v  = einops.rearrange(y, 'bt (s m c) h w -> s bt m (h w) c', s=3, m=num_heads).unbind(0)

# --- Apply flash attention ---
out2 = F.scaled_dot_product_attention(q,k,v)
# out2 = einops.rearrange(out2, '(b s t) m hw c -> (s b) t m hw c', s = 2, b = batch_size)
out2 = einops.rearrange(out2, '(b s t) m (h w) c -> (s b) t (m c) h w', s = 2, b = batch_size, h= height, w=width)
# out2 = einops.rearrange(out2, 'bt m (h w) c -> bt (m c) h w', h=height, w=width)
# out2 = einops.rearrange(out2, '(b s t) c h w -> (s b) t c h w', b=batch_size, s = 2)

diff_std = (out2 - out1).std(dim=(0, 2, 3, 4))
print(diff_std[0]) # it retuns 1e-5!

# %%
