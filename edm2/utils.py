import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
import numpy as np
import einops
from . import misc
#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a:Tensor, b:Tensor, t=0.5):
    if isinstance(t,float):
        return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)
    
    lerp = a + bmult((b - a), t)
    return bmult(lerp, ((1 - t) ** 2 + t ** 2)**(-0.5))

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

        
def bmult(x:Tensor, t:Tensor):
    if t.dim() == 0:
        return t*x
    if t.dim() == 1:
        return einops.einsum(x,t,' b ..., b -> b ...')
    return einops.einsum(x,t,' b c ..., b c-> b c ...')

    
    
# Example implementation
def apply_clipped_grads(model, optimizer, main_loss, adv_loss, max_norm_main, max_norm_adv):
    # Assign to .grad
    optimizer.zero_grad()
    if max_norm_adv is None:
        (main_loss + adv_loss).backward()
        clip_grad_norm_(model.parameters(), max_norm_main)
        return
    
    # Compute gradients
    main_grads = torch.autograd.grad(main_loss, model.parameters(), retain_graph=True)
    adv_grads = torch.autograd.grad(adv_loss, model.parameters())
    
    # Clip gradients (norm clipping)
    clip_grad_norm_(main_grads, max_norm_main)
    clip_grad_norm_(adv_grads, max_norm_adv)
    
    for param, main_g, adv_g in zip(model.parameters(), main_grads, adv_grads):
        param.grad = main_g + adv_g