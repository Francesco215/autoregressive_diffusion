import os
import inspect
import tempfile
from urllib.parse import urlparse

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import einops
from . import misc

class BetterModule(nn.Module):

    def save_to_state_dict(self, path):
        import boto3
        data = {"state_dict": self.state_dict(), "kwargs": self.kwargs}

        if path.startswith("s3://"):
            # Save to a temporary local file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                torch.save(data, tmp.name)
                tmp_path = tmp.name

            # Parse the S3 path
            parsed = urlparse(path)
            bucket_name = parsed.netloc
            key = parsed.path.lstrip("/")

            s3 = boto3.client('s3')
            s3.upload_file(tmp_path, bucket_name, key)
            os.remove(tmp_path)
        else:
            torch.save(data, path)

    @classmethod
    def from_pretrained(cls, checkpoint):
        import boto3
        if isinstance(checkpoint,str):
            if checkpoint.startswith("s3://"):
                # Parse S3 URL
                parsed = urlparse(checkpoint)
                bucket_name = parsed.netloc
                key = parsed.path.lstrip("/")

                # Create /cache directory if not exists
                cache_dir = "/tmp/cache/autoregressive_diffusion_models/"
                os.makedirs(cache_dir, exist_ok=True)

                # Local cache file path
                filename = os.path.basename(key)
                checkpoint = os.path.join(cache_dir, filename)

                # Download from S3 if not already cached
                if not os.path.exists(checkpoint):
                    s3 = boto3.client('s3')
                    s3.download_file(bucket_name, key, checkpoint)
                
            checkpoint = torch.load(checkpoint, weights_only=False)

        model = cls(**checkpoint['kwargs'])

        model.load_state_dict(checkpoint['state_dict'])
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_params(self):
       return sum(p.numel() for p in self.parameters())





    
#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x.to(torch.float32), dim=dim, keepdim=True, dtype=torch.float32)
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

    
    
from torch import nn
from contextlib import contextmanager

def nan_hook(module, input, output):
    """
    Hook function to check for NaNs in the output of a module.
    """
    if isinstance(output, tuple):
        for o in output:
            return nan_hook(module, input, o) 

    if torch.isnan(output).any():
        raise Exception(f"NaN detected in output of {module.__class__.__name__}")


@contextmanager
def nan_inspector(model):
    """
    Context manager to attach NaN-checking hooks to all modules in the model.
    
    Args:
        model (nn.Module): The PyTorch model to inspect.
    
    Usage:
        with nan_inspector(model):
            # Your forward pass code here
    """
    hooks = []
    
    def register_hooks(module):
        # Skip the top-level model itself to avoid redundant checks
        if not isinstance(module, nn.Module) or module == model:
            return
        # Register the hook to monitor the module's forward pass
        hook = module.register_forward_hook(nan_hook)
        hooks.append(hook)
    
    try:
        # Apply the hook registration to all submodules
        model.apply(register_hooks)
        yield  # Run the code inside the 'with' block
    finally:
        # Clean up by removing all hooks
        for hook in hooks:
            hook.remove()

            
def GaussianLoss(mean, logvar, target, eps=1e-4):
    return ((logvar + (mean-target)**2*(torch.exp(-logvar)))*.5+0.918).mean()



def compare_caches(cache1, cache2, rtol=1e-4, atol=1e-4, verbose=True):
    """
    Recursively compares two cache structures (dictionaries, lists, tensors, floats etc.)
    for equality, focusing on PyTorch tensors and Python types.

    Args:
        cache1: The first cache object to compare.
        cache2: The second cache object to compare.
        rtol (float): Relative tolerance for comparing floats and tensors.
                      Defaults to 1e-5.
        atol (float): Absolute tolerance for comparing floats and tensors.
                      Defaults to 1e-8.
        verbose (bool): If True, prints the path of the first detected difference.
                        Defaults to True.

    Returns:
        bool: True if the caches are considered identical within the given
              tolerances, False otherwise.
    """
    return _recursive_compare(cache1, cache2, rtol, atol, verbose, path="cache")

def _recursive_compare(item1, item2, rtol, atol, verbose, path):
    """Helper function for recursive comparison."""

    # 1. Check Type Equality
    if type(item1) is not type(item2):
        if verbose:
            print(f"Type mismatch at {path}: {type(item1)} vs {type(item2)}")
        return False

    # 2. Handle Dictionaries
    if isinstance(item1, dict):
        if item1.keys() != item2.keys():
            if verbose:
                print(f"Dictionary key mismatch at {path}:")
                print(f"  Keys 1 (sorted): {sorted(item1.keys())}")
                print(f"  Keys 2 (sorted): {sorted(item2.keys())}")
                diff1 = set(item1.keys()) - set(item2.keys())
                diff2 = set(item2.keys()) - set(item1.keys())
                if diff1: print(f"  Keys only in cache1: {diff1}")
                if diff2: print(f"  Keys only in cache2: {diff2}")
            return False
        for key in item1:
            new_path = f"{path}[{repr(key)}]"
            # Ensure keys exist before comparing (redundant if keysets match, but safe)
            if key not in item2:
                 if verbose: print(f"Key '{key}' missing in second dict at {path}")
                 return False # Should not happen if keysets matched
            if not _recursive_compare(item1[key], item2[key], rtol, atol, verbose, new_path):
                return False
        return True

    # 3. Handle Lists/Tuples
    elif isinstance(item1, (list, tuple)):
        if len(item1) != len(item2):
            if verbose:
                print(f"Sequence length mismatch at {path}: {len(item1)} vs {len(item2)}")
            return False
        for i in range(len(item1)):
            new_path = f"{path}[{i}]"
            if not _recursive_compare(item1[i], item2[i], rtol, atol, verbose, new_path):
                return False
        return True

    # 4. Handle PyTorch Tensors
    elif isinstance(item1, torch.Tensor):
        if item1.shape != item2.shape:
            if verbose:
                print(f"Tensor shape mismatch at {path}: {item1.shape} vs {item2.shape}")
            return False
        if item1.dtype != item2.dtype:
             if verbose:
                print(f"Tensor dtype mismatch at {path}: {item1.dtype} vs {item2.dtype}")
             # Attempt conversion for comparison if dtypes differ but might be compatible
             try:
                 item2_converted = item2.to(item1.dtype)
             except Exception as e:
                 print(f"  Cannot convert dtype {item2.dtype} to {item1.dtype} for comparison: {e}")
                 return False # Strict dtype check failed and conversion failed
        else:
             item2_converted = item2 # Dtypes match

        # Use torch.allclose for numerical comparison with tolerance
        try:
            # Use device of item1 for comparison if devices differ
            if item1.device != item2_converted.device:
                 item2_converted = item2_converted.to(item1.device)

            # IMPORTANT: Set equal_nan=False by default. If NaNs should compare equal,
            # set it to True when calling compare_caches.
            result = torch.allclose(item1, item2_converted, rtol=rtol, atol=atol, equal_nan=False)
            if not result and verbose:
                print(f"Tensor value mismatch at {path} (using allclose with rtol={rtol}, atol={atol})")
                # Optional: Calculate and print max difference
                try:
                    diff = torch.abs(item1 - item2_converted)
                    max_diff = torch.max(diff).item()
                    print(f"  Max difference: {max_diff:.2e}")
                except Exception:
                    print("  Could not compute max difference.") # Handle potential issues like incompatible types after all
            return result
        except Exception as e:
             if verbose:
                 print(f"Error comparing tensors at {path}: {e}")
             return False # Error during comparison

    # 5. Handle Floats (without NumPy)
    elif isinstance(item1, float):
        abs_diff = abs(item1 - item2)
        # Calculate relative difference carefully to avoid division by zero
        # Use max(abs(item1), abs(item2)) for scale, or a small epsilon if both are near zero
        denominator = max(abs(item1), abs(item2))
        if denominator < atol: # If both numbers are very close to zero, rely on absolute tolerance
            rel_diff = 0.0
        else:
            rel_diff = abs_diff / denominator

        result = (abs_diff <= atol) or (rel_diff <= rtol)

        if not result and verbose:
            print(f"Float value mismatch at {path}: {item1} vs {item2} (atol={atol}, rtol={rtol})")
            print(f"  Absolute Difference: {abs_diff:.2e}, Relative Difference: {rel_diff:.2e}")
        return result

    # 6. Handle Basic Python Types (int, str, bool, None)
    elif isinstance(item1, (int, str, bool)) or item1 is None:
        result = (item1 == item2)
        if not result and verbose:
            print(f"Value mismatch at {path}: {repr(item1)} vs {repr(item2)}")
        return result

    # 7. Handle Other Types (attempt standard equality)
    else:
        try:
            # Be cautious comparing unknown types, might be misleading
            result = (item1 == item2)
            if not result and verbose:
                 print(f"Value mismatch for unrecognized type at {path}: {type(item1)}")
            return result
        except Exception as e:
            # If comparison fails for any reason
            if verbose:
                print(f"Cannot compare items of type {type(item1)} at {path}: {e}")
            return False # Treat as non-equal if comparison raises error