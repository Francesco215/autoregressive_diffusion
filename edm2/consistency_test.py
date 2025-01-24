import torch
import unittest
import logging
import os
from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss

os.environ['TORCH_LOGS']='recompiles'
os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

# Constants
IMG_RESOLUTION = 64
IMG_CHANNELS = 16
N_FRAMES = 8
CUT_FRAME = 3
SEED = 42  # Set a seed for reproducibility

class TestUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
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
            num_blocks=2,
            attn_resolutions=[16, 8]
        ).to("cuda").to(torch.float16)
        print(f"Number of UNet parameters: {sum(p.numel() for p in cls.unet.parameters()) // 1e6}M")

    
    def test_unet_consistency_between_train_and_eval(self):
        # Test consistency between training and evaluation modes
        self.unet.train()
        x = torch.randn(2, 2 * N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=torch.float16)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=torch.float16)
        y_train = self.unet.forward(x, noise_level, None)

        self.unet.eval()
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        noise_level_eval = torch.cat((noise_level[:, :CUT_FRAME], noise_level[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        y_eval = self.unet.forward(x_eval, noise_level_eval, None)

        self.assertTrue(torch.allclose(y_train[:, :CUT_FRAME], y_eval[:, :-1], atol=1e-2))
        self.assertTrue(torch.allclose(y_train[:, CUT_FRAME + N_FRAMES], y_eval[:, -1], atol=1e-2))

    def test_unet_causality(self):
        # make sure that during training the network is fully causal
        self.unet.train()
        x = torch.zeros(2, N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=torch.float16)
        r = torch.randn(2, N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=torch.float16)
        a = torch.cat((x, r), dim=1)
        x[:, CUT_FRAME] = torch.randn(x.shape[0], IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=torch.float16)
        x = torch.cat((x, r), dim=1)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=torch.float16)
        y = self.unet.forward(x, noise_level, None) - self.unet.forward(a, noise_level, None)

        self.assertTrue(torch.allclose(y[:, :CUT_FRAME].std(), torch.tensor(0, dtype=torch.float16), atol=1e-2))
        self.assertFalse(torch.allclose(y[:, CUT_FRAME:N_FRAMES].std(), torch.tensor(0, dtype=torch.float16), atol=1e-2))
        self.assertTrue(torch.allclose(y[:, N_FRAMES:N_FRAMES + CUT_FRAME].std(), torch.tensor(0, dtype=torch.float16), atol=1e-2))
        self.assertFalse(torch.allclose(y[:, N_FRAMES + CUT_FRAME:].std(), torch.tensor(0, dtype=torch.float16), atol=1e-2))

if __name__ == "__main__":
    unittest.main()