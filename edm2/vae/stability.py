import torch
from torch.nn import Module
import numpy as np
from diffusers import AutoencoderKL

class StabilityVAEEncoder(Module):
    def __init__(self,
        vae_name    = 'stabilityai/sd-vae-ft-mse',  # Name of the VAE to use.
        raw_mean    = [5.81, 3.25, 0.12, -2.15],    # Assumed mean of the raw latents.
        raw_std     = [4.17, 4.62, 3.71, 3.28],     # Assumed standard deviation of the raw latents.
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').requires_grad_(False).to("cuda")

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x)['latent_dist']
        return torch.cat([d.mean, d.std], dim=1)

    def _run_vae_decoder(self, x):
        return self._vae.decode(x)['sample']

    def encode_pixels(self, x): # raw pixels => raw latents
        x = x.to(torch.float32) / 255
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

    def encode_latents(self, mean, logvar): # raw latents => final latents
        x = mean + torch.randn_like(mean) * torch.exp(logvar*.5)
        x = x * const_like(x, self.scale).reshape(1, 1, -1, 1, 1)
        x = x + const_like(x, self.bias).reshape(1, 1, -1, 1, 1)
        return x

    def decode(self, x): # final latents => raw pixels
        x = x.to(torch.float32)
        x = x - const_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x


#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)
