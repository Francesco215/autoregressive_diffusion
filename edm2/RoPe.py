import torch
from torch import nn, Tensor
import einops

class RotaryEmbedding(nn.Module):
    # adapted form 
    # https://github.com/lucidrains/PaLM-rlhf-pytorch/blob/6b02ee329106baff78e293afa7d1d2e6dd4e5ca2/palm_rlhf_pytorch/palm.py#L69-L92
    def __init__(self, dim, scale_base = 64):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)


        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("pos_emb_scale", None, persistent=False)

    def make_rotary_embedding(self, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1).to(torch.float16)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** einops.rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1).to(torch.float16)

        freqs, scale = freqs.unsqueeze(1), scale.unsqueeze(1) #this is to avoid the (h w) dim

        return freqs, scale

    def get_rotary_embedding(self,n):
        if (self.pos_emb is not None) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]

        pos_emb, scale = self.make_rotary_embedding(n)
        self.register_buffer("pos_emb", pos_emb.to(torch.float16), persistent=False)
        self.register_buffer("pos_emb_scale", scale.to(torch.float16), persistent=False)
        return pos_emb, scale
    

    def forward(self, q:Tensor, k:Tensor):
        # q,k shape = b m t (h w) c
        
        # split the context subspace from the prediction subspace
        q = einops.rearrange(q, 'b m (a t) i c -> b m a t i c', a=2)
        k = einops.rearrange(k, 'b m (a t) i c -> b m a t i c', a=2)

        pos,scale=self.get_rotary_embedding(q.shape[-3])

        q = (q * pos.cos() + rotate_half(q) * pos.sin())*scale
        k = (k * pos.cos() + rotate_half(k) * pos.sin())/scale

        q = einops.rearrange(q, 'b m a t i c -> b m (a t i) c')
        k = einops.rearrange(k, 'b m a t i c -> b m (a t i) c')
        return q,k

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

