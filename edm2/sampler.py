#%%
import einops
from tqdm import tqdm
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

@torch.no_grad()
def edm_sampler_with_mse(
    net, cache, target=None,  # Added target for MSE calculation
    gnet=None, conditioning=None, num_steps=32, sigma_min=0.002, sigma_max=80, 
    rho=7, guidance=1, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32,
):
    starts_in_training_mode = net.training
    net.eval()
    batch_size, n_frames, channels, height, width = cache.get('shape', (None, None, None, None, None)) # TODO: change this
    device = net.device
    
    # Guided denoiser (same as original)
    def denoise(x, t, cache):
        cache = copy.deepcopy(cache)
        t = torch.ones(batch_size, 1, device=device, dtype=dtype) * t
        
        # cache = cache.copy()
        Dx, cache = net(x, t, conditioning, cache=cache)
        if guidance == 1:
            return Dx, cache
        ref_Dx, _ = net(x, t, conditioning, cache = {}) # TODO: play with the cache
        return ref_Dx.lerp(Dx, guidance), cache

    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
              (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    # Main sampling loop with MSE tracking
    x_next = torch.randn(batch_size, 1, channels, height, width, device=device) * t_steps[0]
    mse_values = []
    mse_pred_values = []
    if target is not None:
        target = target.to(dtype)
        x_next = x_next + target


    net.eval()
    if gnet is not None: gnet.eval()

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Noise injection step
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step
        if i == num_steps:
            x_pred, cache=denoise(x_hat, t_hat, cache)
        else:
            x_pred, _ = denoise(x_hat, t_hat, cache)
        d_cur = (x_hat - x_pred) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction
        if i < num_steps - 1:
            x_pred, _ = denoise(x_next, t_next, cache)
            d_prime = (x_next - x_pred) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Calculate MSE after each step
        if target is not None:
            mse_pred = torch.mean((x_pred - target) ** 2).item()
            mse = torch.mean((x_next - target) ** 2).item()
            mse_values.append(mse)
            mse_pred_values.append(mse_pred)

    if starts_in_training_mode: net.train()
    return x_next, mse_values, mse_pred_values, cache

    
    
    
    
@torch.no_grad()
def sampler_training_callback(latents, precond, autoencoder):
    latents = latents[:,:5]
    # latents = batch["latents"][start:start+num_samples].to(device)
    # text_embeddings = batch["text_embeddings"][start:start+num_samples].to(device)
    context = latents[:, :-1]  # First frames (context)
    target = latents[:, -1:]    # Last frame (ground truth)
    precond.eval()
    sigma = torch.ones(context.shape[:2], device=latents.device) * 0.05
    _, cache = precond(context, sigma)

    # Run sampler with sigma_max=0.5 for initial noise level
    _, mse_steps, mse_pred_values, _ = edm_sampler_with_mse(
        net=precond,
        cache=cache,
        target=target,
        sigma_max=3,  # Initial noise level matches our test
        num_steps=32,
        conditioning=None,
        # gnet=g_net,
        rho = 7,
        guidance = 1,
        S_churn=20,
        S_noise=1,
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(mse_steps, marker='o', linestyle='-', label="MSE")
    plt.plot(mse_pred_values, marker='o', linestyle='-', label="MSE (Predicted)")
    plt.xlabel("Denoising Step")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.title("Denoising Progress (Lower is Better)")
    plt.grid(True)
    plt.legend()
    plt.savefig("images_training/denoising_steps.png")
    plt.close()

    latents = latents[:,:2]
    sigma = torch.ones(latents.shape[:2], device=latents.device) * 0.05
    _, cache = precond(latents, sigma)
    for _ in tqdm(range(4)):
        x, _, _, cache= edm_sampler_with_mse(precond, cache=cache, sigma_max = 80, num_steps=32, rho=7, guidance=1, S_churn=20)
        latents = torch.cat((latents,x),dim=1)

    
    frames = autoencoder.latents_to_frames(latents)

    x = einops.rearrange(frames, 'b (t1 t2) h w c -> b (t1 h) (t2 w) c', t2=8)
    #set high resolution
    plt.imshow(x[0])
    plt.axis('off')
    plt.savefig("images_training/generated_frames.png",bbox_inches='tight',pad_inches=0, dpi=1000)
    plt.close()
    precond.train()
    