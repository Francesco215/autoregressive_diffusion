import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
import einops

from .RoPe import RotaryEmbedding
from .attention_masking import make_train_mask, make_infer_mask, AutoregressiveDiffusionMask
from .utils import  mp_sum
from .conv import MPCausal3DConv, MPConv

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
        self.block_mask = None
        self.training_mask = None        
    

    def forward(self, x:Tensor, batch_size:int):
        if self.num_heads == 0:
            return x

        h, w = x.shape[-2:]
        self.image_size = h*w
        if (self.training_mask is None and self.block_mask is None) or self.last_x_shape != x.shape or self.last_modality != self.training:
            # This can trigger a recompilation of the flex_attention function
            if self.training:
                n_frames = x.shape[0]//(batch_size*2)
                self.training_mask = AutoregressiveDiffusionMask(n_frames, self.image_size)
                self.block_mask = make_train_mask(batch_size, self.num_heads, n_frames, image_size=h*w)
            else:
                n_frames = x.shape[0]//batch_size
                self.block_mask = make_infer_mask(batch_size, self.num_heads, n_frames, image_size=h*w)

            self.last_x_shape = x.shape
            self.last_modality = self.training

        y = self.attn_qkv(x)

        # b:batch, t:time, m: multi-head, s: split, c: channels, h: height, w: width
        y = einops.rearrange(y, '(b t) (s m c) h w -> s b m t (h w) c', b=batch_size, s=3, m=self.num_heads)
        # q, k, v = normalize(y, dim=-1).unbind(0) # pixel norm & split 
        q, k, v = y.unbind(0) # pixel norm & split 

        q, k = self.rope(q, k)

        # i = (h w)
        v = einops.rearrange(v, ' b m t hw c -> b m (t hw) c') # q and k are already rearranged inside of rope

        y = self.flex_attention(q, k, v, self.block_mask)
        y = einops.rearrange(y, 'b m (t h w) c -> (b t) (c m) h w', b=batch_size, h=h, w=w)

        y = self.attn_proj(y)
        return mp_sum(x, y, t=self.attn_balance)
    
    # To log all recompilation reasons, use TORCH_LOGS="recompiles" or torch._logging.set_logs(dynamo=logging.INFO)
    @torch.compile
    def flex_attention(self, q, k, v, block_mask): 
        score_mod = None
        # if block_mask == None:
        #     if self.training:
        #         def causal_mask(score, b, h, q_idx, kv_idx):
        #             return torch.where(self.training_mask(b, h, q_idx, kv_idx), score, -float("inf"))
        #         score_mod = causal_mask
        #     else:
        #         def causal_mask(score, b, h, q_idx, kv_idx):
        #             q_idx, kv_idx = q_idx // self.image_size, kv_idx // self.image_size
        #             return torch.where(q_idx >= kv_idx, score, -float("inf"))
        #         score_mod = causal_mask
        assert score_mod is not None or block_mask is not None, "Either block_mask or score_mod must be defined"
        return flex_attention(q, k, v, score_mod, block_mask)

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