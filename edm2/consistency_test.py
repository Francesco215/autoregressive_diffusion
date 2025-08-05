import einops
import torch
from torch.nn import functional as F
import unittest
import logging
import os
from edm2.attention.attention_modules import FrameAttention
from edm2.conv import MPCausal3DGatedConv
from edm2.attention import VideoAttention
from edm2.networks_edm2 import UNet
from edm2.attention.attention_masking import TrainingMask, make_train_mask
import numpy as np
import random

from edm2.utils import mp_sum, normalize 

# os.environ['TORCH_LOGS']='recompiles'
# os.environ['TORCH_COMPILE_MAX_AUTOTUNE_RECOMPILE_LIMIT']='100000'
# torch._dynamo.config.recompile_limit = 100000
torch._logging.set_logs(dynamo=logging.INFO)

# Constants
IMG_RESOLUTION = 16
BATCH_SIZE = 4
IMG_CHANNELS = 16
N_FRAMES = 8
CUT_FRAME = 3
SEED = 42  # Set a seed for reproducibility
dtype = torch.float32
np.random.seed(42)
random.seed(42)
error_bound = 3e-4


# TODO: make sure that each element of the batch does not interact with any other

class TestConsistencyWithOldAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cls.attention = FrameAttention(channels = 4*IMG_CHANNELS, num_heads = 4).to("cuda").to(dtype)

    def test_attention_consistency_between_frame_and_video_attention(self):
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        y_frame, _ = self.attention(x.clone())

        y = self.attention.attn_qkv(x)
        y = y.reshape(y.shape[0], self.attention.num_heads, -1, 3, y.shape[2] * y.shape[3])
        q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
        w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
        y = torch.einsum('nhqk,nhck->nhcq', w, v)
        y = self.attention.attn_proj(y.reshape(*x.shape))
        y = mp_sum(x, y, t=self.attention.attn_balance)
        

        std_diff = (y-y_frame).std()
        self.assertLessEqual(std_diff.item(), error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}") 


class TestAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cls.attention = VideoAttention(channels = 4*IMG_CHANNELS, num_heads = 4).to("cuda").to(dtype)

    def test_attention_consistency_between_frame_and_video_attention(self):
        self.attention.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        y_video, _ = self.attention(x.clone(), BATCH_SIZE, just_2d=False)
        y_frame, _ = self.attention(x.clone(), BATCH_SIZE, just_2d=True)
        
        y_video = einops.rearrange(y_video, '(b s t) c h w -> s b t c h w', b=BATCH_SIZE, s = 2)
        y_frame = einops.rearrange(y_frame, '(b s t) c h w -> s b t c h w', b=BATCH_SIZE, s = 2)

        std_diff = (y_video-y_frame).std(dim=(0,1,3,4,5))
        self.assertLessEqual(std_diff[0].item(), error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}") 
        self.assertGreaterEqual(std_diff[1:].mean().item(), 1e-2, f"Test failed: std deviation {std_diff} exceeded {error_bound}") 




    def test_attention_consistency_between_flex_and_flash_video_attention(self):
        self.attention.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        y_video, _ = self.attention(x.clone(), BATCH_SIZE, just_2d=False)

        mask = make_train_mask(BATCH_SIZE, 4, N_FRAMES, IMG_RESOLUTION**2) # TODO: check again
        repeated_mask = mask.to_dense()[0,0].bool().repeat_interleave(IMG_RESOLUTION**2,0).repeat_interleave(IMG_RESOLUTION**2,1)

        y = self.attention.attn_qkv(x)
        y = einops.rearrange(y, '(b t) (m c s) h w -> s b m t (h w) c', b=BATCH_SIZE, s=3, m=4)
        q, k, v = normalize(y, dim=-1).unbind(0) # pixel norm & split 

        q, k = self.attention.rope(q, k)
        v = einops.rearrange(v, ' b m t hw c -> b m (t hw) c') # q and k are already rearranged inside of rope
        y = F.scaled_dot_product_attention(q,k,v,repeated_mask)

        y = einops.rearrange(y, 'b m (t h w) c -> (b t) (m c) h w', b=BATCH_SIZE, h=IMG_RESOLUTION, w=IMG_RESOLUTION)
        y = self.attention.attn_proj(y)
        
        y_manual = mp_sum(x, y, t=self.attention.attn_balance)


        std_diff = (y_manual-y_video).std()
        self.assertLessEqual(std_diff.item(), error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")



    def test_attention_consistency_between_train_and_eval(self):
        self.attention.train()
        x = torch.randn(BATCH_SIZE * 2 * N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        y_train, _ = self.attention(x, BATCH_SIZE)
        
        self.attention.eval()
        x = einops.rearrange(x,'(b t) ... -> b t ...', b=BATCH_SIZE)
        x_eval = torch.cat((x[:, :CUT_FRAME], x[:, CUT_FRAME + N_FRAMES].unsqueeze(1)), dim=1)
        x_eval = einops.rearrange(x_eval, 'b t ... -> (b t) ...')
        y_eval, _ = self.attention(x_eval, BATCH_SIZE)

        y_train, y_eval = einops.rearrange(y_train,'(b t) ... -> b t ...', b=BATCH_SIZE), einops.rearrange(y_eval,'(b t) ... -> b t ...', b=BATCH_SIZE)

        std_diff_1 = (y_train[:, :CUT_FRAME] - y_eval[:, :CUT_FRAME]).std().item()
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

        y_cached, cache = self.attention(context, BATCH_SIZE, update_cache=True)
        out, _ = self.attention(last_frame, BATCH_SIZE, cache)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b = BATCH_SIZE)
        out = einops.rearrange(out, '(b t) ... -> b t ...', b = BATCH_SIZE)
        y_cached = torch.cat((y_cached, out), dim=1)

        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")

    def test_attention_consistrency_between_cached_and_non_cached_multistep(self):
        self.attention.eval()
        x = torch.randn(BATCH_SIZE, N_FRAMES, 4*IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        context, second_last, last_frame = x[:, :-2], x[:,-2:-1], x[:,-1:]

        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        context = einops.rearrange(context, 'b t ... -> (b t) ...')
        second_last = einops.rearrange(second_last, 'b t ... -> (b t) ...')
        last_frame = einops.rearrange(last_frame, 'b t ... -> (b t) ...')

        y_non_cached, _ =  self.attention(x, BATCH_SIZE)
        y_cached, cache = self.attention(context, BATCH_SIZE, update_cache=True)
        out1, cache = self.attention(second_last, BATCH_SIZE, cache, update_cache=True)
        out2, _ = self.attention(last_frame, BATCH_SIZE, cache)

        y_non_cached = einops.rearrange(y_non_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)
        y_cached = einops.rearrange(y_cached, '(b t) ... -> b t ...', b=BATCH_SIZE)
        out1 = einops.rearrange(out1, '(b t) ... -> b t ...', b=BATCH_SIZE)
        out2 = einops.rearrange(out2, '(b t) ... -> b t ...', b=BATCH_SIZE)
        y_cached = torch.cat((y_cached, out1, out2), dim=1)

        std_diff = (y_non_cached - y_cached).std().item()
        self.assertLessEqual(std_diff, error_bound, f"Test failed: std deviation {std_diff} exceeded {error_bound}")



    def test_attention_causality(self):
        # make sure that during training the network is fully causal
        self.attention.train()
        x = torch.zeros(BATCH_SIZE, N_FRAMES, 4 * IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        r = torch.randn(BATCH_SIZE, N_FRAMES, 4 * IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        a = torch.cat((x, r), dim=1)
        x[:, CUT_FRAME] = torch.randn(BATCH_SIZE, 4 * IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION, device="cuda", dtype=dtype)
        x = torch.cat((x, r), dim=1)
        a = einops.rearrange(a, 'b t c h w -> (b t) c h w')
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')

        y = self.attention(x, BATCH_SIZE)[0] - self.attention(a, BATCH_SIZE)[0]

        y = einops.rearrange(y, '(b t) c h w -> b t c h w', b = BATCH_SIZE)
        self.assertLessEqual(
            y[:, :CUT_FRAME].std().item(), error_bound,
            f"Test failed: std deviation {y[:, :CUT_FRAME].std()} exceeded {error_bound} before CUT_FRAME"
        )

        # Check that the perturbed frame itself shows a significant difference
        self.assertGreaterEqual(
            y[:, CUT_FRAME:CUT_FRAME+1].std().item(), 0.3,
            f"Test failed: perturbation at CUT_FRAME did not affect output as expected"
        )

        # Check that frames after the perturbation also show change — information flowed forward
        self.assertGreaterEqual( # TODO: CHECK THIS INFORMATION BOTTLENECK
            y[:, CUT_FRAME+1:N_FRAMES].std().item(), 0.3,
            f"Test failed: post-CUT_FRAME frames did not respond to perturbation"
        )

        # For the second half of the sequence (from noise r), check that the frames before CUT_FRAME stay unaffected
        self.assertLessEqual(
            y[:, N_FRAMES:N_FRAMES + CUT_FRAME+1].std().item(), error_bound,
            f"Test failed: std deviation {y[:, N_FRAMES:N_FRAMES + CUT_FRAME+1].std()} exceeded {error_bound} in second sequence before CUT_FRAME"
        )

        # Finally, verify that the perturbation did affect later frames in the second half
        self.assertGreaterEqual( # TODO: CHECK THIS INFORMATION BOTTLENECK
            y[:, N_FRAMES + CUT_FRAME+1:].std().item(), 0.3,
            f"Test failed: post-CUT_FRAME frames in second sequence did not respond to perturbation"
        )


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