#%%
import torch
from torch import profiler
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss
import logging
import os
os.environ['TORCH_LOGS']='recompiles'
os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

img_resolution = 64
img_channels = 16
unet = UNet(img_resolution=img_resolution, # Match your latent resolution
            img_channels=img_channels,
            label_dim = 0,
            model_channels=32,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=2,
            attn_resolutions=[16,8]
            )
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
precond = Precond(unet, use_fp16=False, sigma_data=1.).to("cuda")

# %%

# Generate input tensors
x = torch.randn(4, 16, img_channels, img_resolution, img_resolution, device="cuda")
noise_level = torch.rand(4, 16, device="cuda")
torch.autograd.set_detect_anomaly(True, check_nan=True)
# Profiling
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU, 
        profiler.ProfilerActivity.CUDA
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/root/autoregressive_diffusion/log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(5):  # Run multiple iterations to capture performance over time
        with profiler.record_function("unet"):
            y = precond.forward(x, noise_level, None)
        prof.step()

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


#%%
loss=EDM2Loss()
y=loss(precond, x)

print(y,y.shape)

# %%
#this is to test if the unet is causal



# %%
