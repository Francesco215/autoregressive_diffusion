#%%
from edm2.networks_edm2 import UNet, Precond
import torch
with torch.no_grad():
    device = "cuda"
    my_net = UNet(img_resolution=64, # Match your latent resolution
                img_channels=4, # Match your latent channels
                label_dim = 1000,
                model_channels=128,
                channel_mult=[1,2,3,4],
                channel_mult_noise=None,
                channel_mult_emb=None,
                num_blocks=3,
                video_attn_resolutions=[8],
                frame_attn_resolutions=[16],
                ).to("cuda")
                

    # %%
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/'
    model_name = 'edm2-img512-xs-2147483-0.135.pkl'

    import dnnlib
    import pickle
    with dnnlib.util.open_url(f'{model_name}') as f:
        data = pickle.load(f)
    net= data['ema'].to("cuda")

    # %%

    my_net.load_from_2d(net.unet)
    net.unet.emb_label=None


    # %% #this part o code below it works!
    n_images = 4
    c_noise = torch.randn(1,n_images).to(device).exp()+0.1
    label_dim = 1000
    label = torch.randn(n_images, 512).to(device)
    x = torch.randn(n_images, 5, 16, 16).to(device)
    b1=my_net.enc['64x64_conv'].forward(x, label, 1, c_noise, just_2d=True)
    b2=net.unet.enc['64x64_conv'].forward(x)

    (b1[0]-b2).std() # this works, it's about equal to 0.0001
    #%%

    n_images = 4
    c_noise = torch.randn(1,n_images).to(device).exp()+0.1
    label_dim = 1000
    label = torch.randn(n_images, 512).to(device)
    x = torch.randn(n_images, 256, 16, 16).to(device)

    b1=my_net.enc['16x16_block0'].forward(x, label, 1, c_noise, just_2d=True)
    b2=net.unet.enc['16x16_block0'].forward(x, label)
    (b1[0]-b2).std() # this works, it's about equal to 0.0001

    # %%
    n_images = 4
    x = torch.randn(1, n_images, 4, 64, 64).to(device)
    c_noise = torch.randn(1,n_images).to(device)
    y = my_net(x, c_noise, None, cache=None, update_cache=False, just_2d=True)
    x, c_noise = x[0], c_noise[0]
    y_edm = net.unet(x, c_noise, None)
    (y[0]-y_edm).std() # this doesn't work (it's about equal to 1.2)



# %%
