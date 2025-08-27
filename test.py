#%%
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask
from edm2.attention.attention_masking import TrainingMask, make_train_mask, make_infer_mask
from edm2.attention.attention_modules import compiled_flex_attention

image_width, n_frames = 8, 64
batch_size = 4
num_heads = 4
image_size= image_width **2
channels = 16

y = torch.randn(3, batch_size, n_frames, num_heads, image_size, channels)

y[:,:,-1]/=0

y=einops.rearrange(y, 's b t m n c -> s b m (t n) c').to("cuda")
q, k, v = y.unbind(0)

infer_score, infer_mask = make_infer_mask(batch_size, num_heads, n_frames, image_size)
out_i = compiled_flex_attention(q,k,v, infer_score, infer_mask)
out_i = einops.rearrange(out_i, 'b m (t n) c -> b t m n c', t=n_frames)
print(out_i.mean(dim=(0,2,3,4)))
# out_i = compiled_flex_attention(q,k,v, infer_score, infer_mask)

# %%

y = torch.randn(3, batch_size, n_frames*2, num_heads, image_size, channels, device = "cuda")
y[:,:,2]/=0
y=einops.rearrange(y, 's b t m n c -> s b m (t n) c').to("cuda")
q, k, v = y.unbind(0)

train_score, train_mask = make_train_mask(batch_size, num_heads, n_frames, image_size)
out_t = compiled_flex_attention(q,k,v, train_score, train_mask)
out_t = einops.rearrange(out_t, 'b m (t n) c -> b t m n c', t=n_frames*2)
print(out_t.mean(dim=(0,2,3,4)))
# %%
