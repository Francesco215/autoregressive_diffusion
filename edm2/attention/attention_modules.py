import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention

import einops

from .RoPe import RotaryEmbedding
from .attention_masking import make_train_mask, make_infer_mask
from ..utils import  mp_sum
from ..conv import MPConv

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
    

    def forward(self, x:Tensor, batch_size:int, cache:Tensor=None, update_cache=False, just_2d=False):
        if self.num_heads == 0:
            return x, None

        h, w = x.shape[-2:]
        y = self.attn_qkv(x)

        if just_2d: #just use the code from frame attention
            y = einops.rearrange(y, 'bt (s m c) h w -> s bt m (h w) c', s=3, m=self.num_heads)
            q, k, v =y.unbind(0)

            y = F.scaled_dot_product_attention(q, k, v)
            y = einops.rearrange(y, 'bt m (h w) c -> bt (m c) h w', h=h, w=w)

            y = self.attn_proj(y)
            return mp_sum(x, y, t=self.attn_balance), cache
            
        # b:batch, t:time, m: multi-head, s: split, c: channels, h: height, w: width
        y = einops.rearrange(y, '(b t) (s m c) h w -> s b m t (h w) c', b=batch_size, s=3, m=self.num_heads)
        q, k, v = y.unbind(0) # pixel norm & split 

        if not self.training: # Handling of the cache during inference
            if cache is not None:
                # TODO: check if we need to clone the tensors to avoid modification of the cache
                # TODO: check if it's better to load x and eval the QK matrices (it could reduce the bandwidth required)
                cached_k, cached_v = cache 
                k, v = torch.cat((cached_k.clone(), k), dim=-3), torch.cat((cached_v.clone(), v), dim=-3)
            if update_cache: cache = (k, v)

        q, k = self.rope(q, k)
        # q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        v = einops.rearrange(v, ' b m t hw c -> b m (t hw) c') # q and k are already rearranged inside of rope

        if self.training:
            n_frames, image_size= x.shape[0]//(batch_size*2), h*w
            train_mask = make_train_mask(batch_size, self.num_heads, n_frames, image_size)
            y = compiled_flex_attention(q, k, v, block_mask=train_mask)

        else:
            if q.shape[-2] == h*w: # if only one frame is being generated 
                y = F.scaled_dot_product_attention(q, k, v)

            elif q.shape == k.shape: # if we are evaluating the network forward pass of the context frames
                n_frames = q.shape[-2]//(h*w)
                score_mod, inference_mask = make_infer_mask(batch_size, self.num_heads, n_frames, h*w)
                y = compiled_flex_attention(q, k, v, score_mod, inference_mask)
            else:
                raise NotImplementedError("The inference mask is not implemented for this case")

        y = einops.rearrange(y, 'b m (t h w) c -> (b t) (c m) h w', b=batch_size, h=h, w=w)
        y = self.attn_proj(y)
        
        return mp_sum(x, y, t=self.attn_balance), cache
    
# To log all recompilation reasons, use TORCH_LOGS="recompiles" or torch._logging.set_logs(dynamo=logging.INFO)
@torch.compile
def compiled_flex_attention(q, k, v, score_mod=None, block_mask=None):
    assert score_mod is not None or block_mask is not None
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        
        
        
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

    def forward(self, x, batch_size, cache=None, update_cache=False, just_2d=True):
        if self.num_heads==0:
            return x, None
        # x.shape = bt c h w
        h, w = x.shape[-2:]
        y = self.attn_qkv(x)

        y = einops.rearrange(y, 'bt (s m c) h w -> s bt m (h w) c', s=3, m=self.num_heads)
        q, k, v =y.unbind(0)

        y = F.scaled_dot_product_attention(q, k, v)
        y = einops.rearrange(y, 'bt m (h w) c -> bt (m c) h w', h=h, w=w)

        y = self.attn_proj(y)
        return mp_sum(x, y, t=self.attn_balance), None