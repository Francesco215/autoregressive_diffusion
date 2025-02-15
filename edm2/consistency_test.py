import einops
import torch
import unittest
import logging
import os
from edm2.conv import MPCausal3DConv
from edm2.attention import VideoAttention
from edm2.networks_edm2 import UNet
import numpy as np
import random 

os.environ['TORCH_LOGS']='recompiles'
os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

# Constants
IMG_RESOLUTION = 64
BATCH_SIZE = 16
IMG_CHANNELS = 16
N_FRAMES = 8
CUT_FRAME = 3
SEED = 42  # Set a seed for reproducibility
dtype = torch.float32
np.random.seed(42)
random.seed(42)
error_bound = 4e-4
class TestUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Initialize UNet and Precond
        cls.unet = UNet(
            img_resolution=IMG_RESOLUTION,
            img_channels=IMG_CHANNELS,
            label_dim=0,
            model_channels=32,
            channel_mult=[1, 2, 2, 4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=3,
            attn_resolutions=[16,8]
        ).to("cuda").to(dtype)
        print(f"Number of UNet parameters: {sum(p.numel() for p in cls.unet.parameters()) // 1e6}M")

        cls.attention = VideoAttention(channels = 4*IMG_CHANNELS, num_heads = 4).to("cuda").to(dtype)
        cls.conv3d = MPCausal3DConv(IMG_CHANNELS, IMG_CHANNELS, kernel = (3,3,3)).to("cuda").to(dtype)

    def test_attention_consistency_between_train_and_eval(self):
        self.attention.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=dtype)
        y_train, None = self.attention(x, BATCH_SIZE)
        
        self.attention.eval()
        x = einops.rearrange(x,'(b l) ... -> b l ...', b=BATCH_SIZE)
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        x_eval = einops.rearrange(x_eval, 'b l ... -> (b l ) ...')
        y_eval = self.attention(x_eval, BATCH_SIZE)

        y_train, y_eval = einops.rearrange(y_train,'(b l) ... -> b l ...', b=BATCH_SIZE), einops.rearrange(y_eval,'(b l) ... -> b l ...', b=BATCH_SIZE)

        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :-1]).std().item()
        std_diff_2 = (y_train[:, CUT_FRAME + N_FRAMES] - y_eval[:, -1]).std().item()

        self.assertLessEqual(std_diff_1, error_bound, f"Test failed: std deviation {std_diff_1} exceeded {error_bound}")
        self.assertLessEqual(std_diff_2, error_bound, f"Test failed: std deviation {std_diff_2} exceeded {error_bound}")


    def test_3d_conv_consistency_between_train_and_eval(self):
        self.conv3d.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=dtype)
        y_train = self.conv3d(x, BATCH_SIZE)
        
        self.conv3d.eval()
        x = einops.rearrange(x,'(b l) ... -> b l ...', b=BATCH_SIZE)
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        x_eval = einops.rearrange(x_eval, 'b l ... -> (b l ) ...')
        y_eval = self.conv3d(x_eval, BATCH_SIZE)

        y_train, y_eval = einops.rearrange(y_train,'(b l) ... -> b l ...', b=BATCH_SIZE), einops.rearrange(y_eval,'(b l) ... -> b l ...', b=BATCH_SIZE)

        std_diff_0 = (y_train[:, :1] - y_eval[:, :1]).std().item()
        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :-1]).std().item()
        std_diff_2 = (y_train[:, CUT_FRAME + N_FRAMES] - y_eval[:, -1]).std().item()

        self.assertLessEqual(std_diff_0, error_bound, f"Test failed: std deviation {std_diff_0} exceeded {error_bound}")
        self.assertLessEqual(std_diff_1, error_bound, f"Test failed: std deviation {std_diff_1} exceeded {error_bound}")
        self.assertLessEqual(std_diff_2, error_bound, f"Test failed: std deviation {std_diff_2} exceeded {error_bound}")
    
    def test_unet_consistency_between_train_and_eval(self):
        # Test consistency between training and evaluation modes
        self.unet.train()
        x = torch.randn(2, 2 * N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=dtype)
        y_train = self.unet.forward(x, noise_level, None)

        self.unet.eval()
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        noise_level_eval = torch.cat((noise_level[:, :CUT_FRAME], noise_level[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        y_eval = self.unet.forward(x_eval, noise_level_eval, None)
        
        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :-1]).std().item()
        std_diff_2 = (y_train[:, CUT_FRAME + N_FRAMES] - y_eval[:, -1]).std().item()

        self.assertLessEqual(std_diff_1, error_bound, f"Test failed: std deviation {std_diff_1} exceeded {error_bound}")
        self.assertLessEqual(std_diff_2, error_bound, f"Test failed: std deviation {std_diff_2} exceeded {error_bound}")


    def test_unet_causality(self):
        # make sure that during training the network is fully causal
        self.unet.train()
        x = torch.zeros(2, N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        r = torch.randn(2, N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        a = torch.cat((x, r), dim=1)
        x[:, CUT_FRAME] = torch.randn(x.shape[0], IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        x = torch.cat((x, r), dim=1)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=dtype)
        y = self.unet.forward(x, noise_level, None) - self.unet.forward(a, noise_level, None)

        self.assertTrue((y[:, :CUT_FRAME].std()<error_bound).item())
        self.assertTrue((y[:, CUT_FRAME:N_FRAMES].std()>0.3).item())
        self.assertTrue((y[:, N_FRAMES:N_FRAMES + CUT_FRAME].std()<error_bound).item())
        self.assertFalse((y[:, N_FRAMES + CUT_FRAME:].std()>0.3).item())

if __name__ == "__main__":
    unittest.main()