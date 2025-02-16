import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention

import einops

from .RoPe import RotaryEmbedding
from .attention_masking import make_train_mask, make_infer_mask
from .utils import  mp_sum
from .conv import MPConv

#----------------------------------------------------------------------------
# Self-Attention module. It shouldn't need anything extra to make it magniture-preserving
class VideoAttention(nn.Module):
    def __init__(self, channels, num_heads, attn_balance = 0.3):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attn_balance = attn_balance
        if num_heads == 0:
            return

        self.attn_qkv = MPConv(channels, channels * 3, kernel=[1,1]) 
        self.attn_proj = MPConv(channels, channels, kernel=[1,1]) 
        self.rope = RotaryEmbedding(channels//num_heads)
        self.train_mask = None
    

    def forward(self, x:Tensor, batch_size:int, cache:Tensor=None):
        if self.num_heads == 0:
            return x, None

        h, w = x.shape[-2:]
        if self.training and ((self.training_mask is None and self.train_mask is None) or self.last_x_shape != x.shape):
            # This can trigger a recompilation of the flex_attention function
            n_frames, image_size= x.shape[0]//(batch_size*2), h*w
            self.train_mask = make_train_mask(batch_size, self.num_heads, n_frames, image_size)
            self.last_x_shape = x.shape

        y = self.attn_qkv(x)

        # b:batch, t:time, m: multi-head, s: split, c: channels, h: height, w: width
        y = einops.rearrange(y, '(b t) (s m c) h w -> s b m t (h w) c', b=batch_size, s=3, m=self.num_heads)
        q, k, v = y.unbind(0) # pixel norm & split 

        if not self.training:
            if cache is not None:
                cached_k, cached_v = cache # TODO: check if we need to clone the tensors to avoid modification of the cache
                k, v = torch.cat((cached_k, k), dim=-3), torch.cat((cached_v, v), dim=-3)
            # TODO: this can be optimized because you only need to update the cache only at the last diffusion step
            # but maybe since i'm just updating the pointer it could not be a big deal
            cache = (k, v)

        q, k = self.rope(q, k)
        v = einops.rearrange(v, ' b m t hw c -> b m (t hw) c') # q and k are already rearranged inside of rope

        if self.training:
            y = self.flex_attention(q, k, v, self.train_mask)
        else:
            if q.shape[-2] == h*w:
                y = F.scaled_dot_product_attention(q, k, v)
                # attention = F.softmax(q @ k.transpose(-2,-1), dim=-1)
                # y = attention @ v
            elif q.shape == k.shape:
                n_frames = q.shape[-2]//(h*w)
                inference_mask = make_infer_mask(batch_size, self.num_heads, n_frames, h*w)
                y = self.infer_flex_attention(q, k, v, inference_mask)
            else:
                raise NotImplementedError("The inference mask is not implemented for this case")

        y = einops.rearrange(y, 'b m (t h w) c -> (b t) (c m) h w', b=batch_size, h=h, w=w)
        y = self.attn_proj(y)
        
        return mp_sum(x, y, t=self.attn_balance), cache
    
    # To log all recompilation reasons, use TORCH_LOGS="recompiles" or torch._logging.set_logs(dynamo=logging.INFO)
    @torch.compile
    def flex_attention(self, q, k, v, block_mask): 
        return flex_attention(q, k, v, block_mask=block_mask)

    # @torch.jit.script
    def infer_flex_attention(self, q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

        
        
        
class FrameAttention(nn.Module):
    def __init__(self, channels, num_heads, attn_balance = 0.3):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attn_balance = attn_balance
        if num_heads == 0:
            return

        self.attn_qkv = MPConv(channels, channels * 3, kernel=[1,1]) 
        self.attn_proj = MPConv(channels, channels, kernel=[1,1]) 

    def forward(self, x, batch_size):
        if self.num_heads==0:
            return x
        # x.shape = bt c h w
        h, w = x.shape[-2:]
        y = self.attn_qkv(x)

        y = einops.rearrange(y, 'bt (s m c) h w -> s bt m (h w) c', s=3, m=self.num_heads)
        q, k, v =y.unbind(0)

        y = F.scaled_dot_product_attention(q, k, v)
        y = einops.rearrange(y, 'bt m (h w) c -> bt (m c) h w', h=h, w=w)

        y = self.attn_proj(y)
        return mp_sum(x, y, t=self.attn_balance)