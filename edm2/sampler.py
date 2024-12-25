#%%
import torch
import numpy as np

@torch.no_grad()
def edm_sampler(
    net,                         # Main network. Path, URL, or torch.nn.Module.
    context,                     # Input tensor. context.shape = b t c h w
    gnet = None,                 # Guiding network. None = same as main network.
    labels=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    context = context.to(dtype)
    batch_size, n_frames, channels, height, width = context.shape
    # Guided denoiser.
    def denoise(x, t):
        context_t = torch.ones(batch_size,n_frames,device=t.device,dtype=dtype)*0.1
        t = torch.ones(batch_size,1,device=t.device,dtype=dtype)*t
        t = torch.cat((context_t,t),dim=1)
        
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=context.device) 
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho 
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0    
    
    # Main sampling loop.
    noise = torch.randn(batch_size, 1, channels, height, width, device=context.device, dtype=dtype) * t_steps[0]
    x_next = torch.cat((context, noise), dim=1)

    net.eval()
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        x_cur[:,:-1] = context.clone()

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_next[:,:-1] = context.clone()
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    net.train()
    return x_next
