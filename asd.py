#%%
import torch
import numpy as np
from torch.autograd import Function

###############################################################################
# CUSTOM FUNCTION
###############################################################################
from edm2.conv import Weight, EfficientWeight

###############################################################################
# TESTS
###############################################################################
import torch
import time

def measure_speed_and_utilization():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")


    in_channels = 512
    out_channels = 512
    image_size = 16
    kernel = (3, 3)
    gain = 1.0

    dummy_input = torch.randn(out_channels, in_channels, *kernel, device=device)  # Example input

    old_module = Weight(in_channels, out_channels, kernel).to(device)
    fast_module = EfficientWeight(in_channels, out_channels, kernel).to(device)

    with torch.no_grad():
        old_module.weight.copy_(fast_module.weight.clone())

    old_module.train()
    fast_module.train()

    # Warmup CUDA (to remove initialization overhead)
    for _ in range(5):
        old_module(dummy_input, gain=gain)
        fast_module(dummy_input, gain=gain)

    # -------------------------------
    # Measure Forward Pass Speed
    # -------------------------------
    torch.cuda.synchronize()
    start_old = time.perf_counter()
    out_old = old_module(dummy_input, gain=gain)
    torch.cuda.synchronize()
    old_forward_time = time.perf_counter() - start_old

    torch.cuda.synchronize()
    start_fast = time.perf_counter()
    out_fast = fast_module(dummy_input, gain=gain)
    torch.cuda.synchronize()
    fast_forward_time = time.perf_counter() - start_fast

    print(f"Forward Pass - Old: {old_forward_time:.6f}s, Fast: {fast_forward_time:.6f}s")

    # -------------------------------
    # Measure Backward Pass Speed
    # -------------------------------
    loss_old = out_old.pow(2).sum()
    loss_fast = out_fast.pow(2).sum()

    old_module.weight.grad = None
    fast_module.weight.grad = None

    torch.cuda.synchronize()
    start_old = time.perf_counter()
    loss_old.backward(retain_graph=True)
    torch.cuda.synchronize()
    old_backward_time = time.perf_counter() - start_old

    torch.cuda.synchronize()
    start_fast = time.perf_counter()
    loss_fast.backward()
    torch.cuda.synchronize()
    fast_backward_time = time.perf_counter() - start_fast

    print(f"Backward Pass - Old: {old_backward_time:.6f}s, Fast: {fast_backward_time:.6f}s")
    speed_multiplier = old_backward_time / fast_backward_time
    print(f"Speed Multiplier: {speed_multiplier:.2f}x")

    # -------------------------------
    # Measure CUDA Memory Usage
    # -------------------------------
    old_mem = torch.cuda.memory_allocated()
    fast_mem = torch.cuda.memory_reserved()

    print(f"CUDA Memory - old: {old_mem / 1e6:.2f}MB, fast: {fast_mem / 1e6:.2f}MB")

    # -------------------------------
    # Check Output & Gradient Similarity
    # -------------------------------
    print("Forward pass outputs close?", torch.allclose(out_old, out_fast, rtol=1e-4, atol=1e-4))

    grad_old = old_module.weight.grad
    grad_fast = fast_module.weight.grad

    print("Backward pass gradients close?", torch.allclose(grad_old, grad_fast, rtol=1e-4, atol=1e-4))

if __name__ == "__main__":
    measure_speed_and_utilization()

# %%
