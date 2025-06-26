import einops
import torch
import unittest
import logging
import os
from edm2.conv import MPCausal3DGatedConv
from edm2.attention import VideoAttention
from edm2.unet import UNet
import numpy as np
import random 

# os.environ['TORCH_LOGS']='recompiles'
# os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
# torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

# Constants
IMG_RESOLUTION = 64
BATCH_SIZE = 4
IMG_CHANNELS = 16
N_FRAMES = 8
CUT_FRAME = 3
SEED = 42  # Set a seed for reproducibility
dtype = torch.float32
np.random.seed(42)
random.seed(42)
error_bound = 1e-2
class TestAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cls.attention = VideoAttention(channels = 4*IMG_CHANNELS, num_heads = 4).to("cuda").to(dtype)

    def test_attention_consistency_between_train_and_eval(self):
        self.attention.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        y_train, _ = self.attention(x, BATCH_SIZE)
        
        self.attention.eval()
        x = einops.rearrange(x,'(b l) ... -> b l ...', b=BATCH_SIZE)
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        x_eval = einops.rearrange(x_eval, 'b l ... -> (b l ) ...')
        y_eval, _ = self.attention(x_eval, BATCH_SIZE)

        y_train, y_eval = einops.rearrange(y_train,'(b l) ... -> b l ...', b=BATCH_SIZE), einops.rearrange(y_eval,'(b l) ... -> b l ...', b=BATCH_SIZE)

        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :-1]).std().item()
        std_diff_2 = (y_train[:, CUT_FRAME + N_FRAMES] - y_eval[:, -1]).std().item()

        self.assertLessEqual(std_diff_1, error_bound, f"Test failed: std deviation {std_diff_1} exceeded {error_bound}")
        self.assertLessEqual(std_diff_2, error_bound, f"Test failed: std deviation {std_diff_2} exceeded {error_bound}")

    def test_attention_consistrency_between_cached_and_non_cached(self):
        self.attention.eval()
        x = torch.randn(BATCH_SIZE, N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        context, last_frame = x[:, :-1], x[:,-1:]
        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')
        y_non_cached, _ =  self.attention(x, BATCH_SIZE)
        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)

        y_cached, cache = self.attention(context, BATCH_SIZE)
        out, _ = self.attention(last_frame, BATCH_SIZE, cache)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b = BATCH_SIZE)
        out = einops.rearrange(out, '(b t) ... -> b t ...', b = BATCH_SIZE)
        y_cached = torch.cat((y_cached, out), dim=1)

        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")

    def test_attention_consistrency_between_cached_and_non_cached_multistep(self):
        self.attention.eval()
        b = 1
        img_res = 16
        x = torch.randn(b, N_FRAMES, 4*IMG_CHANNELS, img_res, img_res, device="cuda", dtype=dtype)
        context, second_last, last_frame = x[:, :-2], x[:,-2:-1], x[:,-1:]

        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        second_last = einops.rearrange(second_last, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')

        y_non_cached, _ =  self.attention(x, b)
        y_cached, cache = self.attention(context, b, update_cache=True)
        out1, cache = self.attention(second_last, b, cache, update_cache=True)
        out2, _ = self.attention(last_frame, b, cache)

        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=b)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b=b)
        out1 = einops.rearrange(out1, '(b t) ... -> b t ...', b=b)
        out2 = einops.rearrange(out2, '(b t) ... -> b t ...', b=b)
        y_cached = torch.cat((y_cached, out1, out2), dim=1)

        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")

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
            video_attn_resolutions=[16,8]
        ).to("cuda").to(dtype)
        print(f"Number of UNet parameters: {sum(p.numel() for p in cls.unet.parameters()) // 1e6}M")

    ## from this point onward the tests are obsolete so they might 
    def test_unet_consistency_between_train_and_eval(self):
        # Test consistency between training and evaluation modes
        self.unet.train()
        x = torch.randn(2, 2 * N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        noise_level = torch.zeros(x.shape[:2], device="cuda", dtype=dtype)
        y_train, _ = self.unet(x, noise_level, conditioning=None)

        self.unet.eval()
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        noise_level_eval = torch.cat((noise_level[:, :CUT_FRAME], noise_level[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        y_eval, _ = self.unet(x_eval, noise_level_eval, conditioning=None) # if i dont pass explicitly the last argument cache={} it doesn't work despite the fact that it is the default argument!
        
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
        y = self.unet(x, noise_level, None)[0] - self.unet(a, noise_level, None)[0]

        self.assertLessEqual(y[:, :CUT_FRAME].std().item(),error_bound, f"Test failed: std deviation {y[:, :CUT_FRAME].std()} exceeded {error_bound}")
        self.assertTrue((y[:, CUT_FRAME:N_FRAMES].std()>0.3).item())
        self.assertLessEqual(y[:, N_FRAMES:N_FRAMES + CUT_FRAME].std().item(),error_bound, f"Test failed: std deviation {y[:, N_FRAMES:N_FRAMES + CUT_FRAME].std()} exceeded {error_bound}")
        self.assertTrue((y[:, N_FRAMES + CUT_FRAME:].std()>0.3).item())


class TestMPCausal3DConv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cls.conv3d = MPCausal3DConv(IMG_CHANNELS, IMG_CHANNELS, kernel=(3,3,3)).to("cuda").to(dtype)
    
    def test_conv_consistency_between_train_and_eval(self):
        self.conv3d.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        y_train, _ = self.conv3d(x, None, BATCH_SIZE)
        
        self.conv3d.eval()
        x = einops.rearrange(x, '(b l) ... -> b l ...', b=BATCH_SIZE)
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        x_eval = einops.rearrange(x_eval, 'b l ... -> (b l ) ...')
        y_eval, _ = self.conv3d(x_eval, None, BATCH_SIZE)
        
        y_train, y_eval = einops.rearrange(y_train,'(b l) ... -> b l ...', b=BATCH_SIZE), einops.rearrange(y_eval,'(b l) ... -> b l ...', b=BATCH_SIZE)
        
        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :-1]).std().item()
        std_diff_2 = (y_train[:, CUT_FRAME + N_FRAMES] - y_eval[:, -1]).std().item()
        
        self.assertLessEqual(std_diff_1, error_bound, f"Test failed: std deviation {std_diff_1} exceeded {error_bound}")
        self.assertLessEqual(std_diff_2, error_bound, f"Test failed: std deviation {std_diff_2} exceeded {error_bound}")
    
    def test_conv_consistency_between_cached_and_non_cached(self):
        self.conv3d.eval()
        x = torch.randn(BATCH_SIZE, N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        context, last_frame = x[:, :-1], x[:,-1:]
        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')
        
        y_non_cached, _ = self.conv3d(x, None, BATCH_SIZE)
        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)
        
        y_cached, cache = self.conv3d(context, None, BATCH_SIZE)
        out, _ = self.conv3d(last_frame, None, BATCH_SIZE, cache=cache)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)
        out = einops.rearrange(out, '(b t) ... -> b t ...', b=BATCH_SIZE)
        y_cached = torch.cat((y_cached, out), dim=1)
        
        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")
    
    def test_conv_consistency_between_cached_and_non_cached_multistep(self):
        self.conv3d.eval()
        b = 1
        img_res = 16
        x = torch.randn(b, N_FRAMES, IMG_CHANNELS, img_res, img_res, device="cuda", dtype=dtype)
        context, second_last, last_frame = x[:, :-2], x[:,-2:-1], x[:,-1:]
        
        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        second_last = einops.rearrange(second_last, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')
        
        y_non_cached, _ = self.conv3d(x, None, b)
        y_cached, cache = self.conv3d(context, None, b)
        out1, cache = self.conv3d(second_last, None, b, cache=cache)
        out2, _ = self.conv3d(last_frame, None, b, cache=cache)
        
        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=b)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b=b)
        out1 = einops.rearrange(out1, '(b t) ... -> b t ...', b=b)
        out2 = einops.rearrange(out2, '(b t) ... -> b t ...', b=b)
        y_cached = torch.cat((y_cached, out1, out2), dim=1)
        
        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")

        
        
class TestMPCausal3DGatedConv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cls.conv3d = MPCausal3DGatedConv(IMG_CHANNELS, IMG_CHANNELS, kernel=(3,3,3)).to("cuda").to(dtype)
    
    def test_conv_consistency_between_train_and_eval(self):
        self.conv3d.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        c_noise = torch.randn(BATCH_SIZE, 2 * N_FRAMES, device =x.device)
        y_train, _ = self.conv3d(x, None, BATCH_SIZE, c_noise)
        
        self.conv3d.eval()
        x = einops.rearrange(x, '(b l) ... -> b l ...', b=BATCH_SIZE)
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        x_eval = einops.rearrange(x_eval, 'b l ... -> (b l) ...')
        # c_noise = einops.rearrange(c_noise, '(b t) -> b t')
        c_noise = torch.cat((c_noise[:, :CUT_FRAME], c_noise[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        y_eval, _ = self.conv3d(x_eval, None, BATCH_SIZE, c_noise)
        
        y_train, y_eval = einops.rearrange(y_train,'(b l) ... -> b l ...', b=BATCH_SIZE), einops.rearrange(y_eval,'(b l) ... -> b l ...', b=BATCH_SIZE)
        
        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :-1]).std().item()
        std_diff_2 = (y_train[:, CUT_FRAME + N_FRAMES] - y_eval[:, -1]).std().item()
        
        self.assertLessEqual(std_diff_1, error_bound, f"Test failed: std deviation {std_diff_1} exceeded {error_bound}")
        self.assertLessEqual(std_diff_2, error_bound, f"Test failed: std deviation {std_diff_2} exceeded {error_bound}")
    
    def test_conv_consistency_between_cached_and_non_cached(self):
        self.conv3d.eval()
        x = torch.randn(BATCH_SIZE, N_FRAMES, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        c_noise = torch.randn(BATCH_SIZE, N_FRAMES, device = x.device)
        context, last_frame = x[:, :-1], x[:,-1:]
        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')
        
        y_non_cached, _ = self.conv3d(x, None, BATCH_SIZE, c_noise)
        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)
        
        y_cached, cache = self.conv3d(context, None, BATCH_SIZE, c_noise[:,:-1], update_cache=True)
        out, _ = self.conv3d(last_frame, None, BATCH_SIZE, c_noise[:,-1:], cache=cache)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)
        out = einops.rearrange(out, '(b t) ... -> b t ...', b=BATCH_SIZE)
        y_cached = torch.cat((y_cached, out), dim=1)
        
        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")
    
    def test_conv_consistency_between_cached_and_non_cached_multistep(self):
        self.conv3d.eval()
        b = BATCH_SIZE
        img_res = 16
        x = torch.randn(b, N_FRAMES, IMG_CHANNELS, img_res, img_res, device="cuda", dtype=dtype)
        c_noise = torch.randn(b, N_FRAMES, device = x.device)
        context, second_last, last_frame = x[:, :-2], x[:,-2:-1], x[:,-1:]
        
        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        second_last = einops.rearrange(second_last, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')
        
        y_non_cached, _ = self.conv3d(x, None, b, c_noise)
        y_cached, cache = self.conv3d(context, None, b, c_noise[:,:-2], update_cache=True)
        out1, cache = self.conv3d(second_last, None, b, c_noise[:,-2:-1], cache=cache, update_cache=True)
        out2, _ = self.conv3d(last_frame, None, b, c_noise[:,-1:], cache=cache)
        
        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=b)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b=b)
        out1 = einops.rearrange(out1, '(b t) ... -> b t ...', b=b)
        out2 = einops.rearrange(out2, '(b t) ... -> b t ...', b=b)
        y_cached = torch.cat((y_cached, out1, out2), dim=1)
        
        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")

        
        


if __name__ == "__main__":
    unittest.main()