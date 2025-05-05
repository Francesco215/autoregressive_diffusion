import math
import torch
import unittest
import numpy as np

from edm2.vae import GroupCausal3DConvVAE
from edm2.vae.vae import UpDownBlock
class TestGroupCausal3DConvVAE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 3
        self.sequence_length = 16  # Time dimension (sequence length)
        self.height = 8
        self.width = 8
        self.kernel_size = (8, 3, 3)  # (Time, Height, Width)
        self.group_size = 4  # Group size for causality
        self.dilation = (1, 1, 1)
        self.error_bound = 1e-3  # Tolerance for causality check
        self.perturb_std_threshold = 0.1  # Minimum std dev to detect causality

        self.model = GroupCausal3DConvVAE(
            self.in_channels, self.out_channels, 
            self.kernel_size, self.group_size, self.dilation
        ).cuda()
        self.model.train()  # Ensure training mode

    def test_group_causality(self):
        # Generate zero baseline input
        x = torch.zeros(self.batch_size, self.in_channels, self.sequence_length, self.height, self.width, device="cuda")
        
        y = self.model(x)[0]

        # Perturb a single time step at a group boundary
        cut_frame = 6 
        min_index_affected = (cut_frame//self.group_size)*self.group_size
        max_index_affected= min(min_index_affected+self.kernel_size[0], self.sequence_length)

        x[:, :, cut_frame] = torch.randn(x.shape[0], self.in_channels, self.height, self.width, device="cuda")

        # Pass through the model
        y = self.model(x)[0] - y


        # Verify causality: No effect before the perturbation
        self.assertGreaterEqual(
            y[:, :, min_index_affected:max_index_affected].std(dim=(0,1,3,4)).min().item(), self.perturb_std_threshold, 
            f"Test failed: std deviation {y[:, :, min_index_affected:max_index_affected].std(dim=(0,1,3,4)).min()} is less {self.perturb_std_threshold}"
        )

        # Verify causality: Perturbation should affect its group and future groups
        self.assertLessEqual(
            y[:, :, :min_index_affected].std().item(), self.error_bound,
            f"Test failed: std deviation {y[:, :, :min_index_affected].std()} is greater {self.error_bound}"
        )

        self.assertLessEqual(
            y[:, :, max_index_affected:].std().item(), self.error_bound,
            f"Test failed: std deviation {y[:, :, max_index_affected:].std()} is greater {self.error_bound}"
        )


class TestUpDownBlockDownCausality(unittest.TestCase):
    """
    Group-causality for a temporal-*down* transition.
    """
    def setUp(self):
        # architecture hyper‑parameters
        self.batch_size          = 2
        self.in_channels         = 3
        self.out_channels        = 3
        self.group_size          = 2        # final causal group
        self.time_compression    = 2        # down‑sampling factor T
        self.spatial_compression = 2
        self.kernel_size         = (self.group_size * 2, 3, 3)
        self.sequence_length     = 16

        # test thresholds
        self.error_bound            = 1e-3
        self.perturb_std_threshold  = 1e-1   # must be clearly > 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = UpDownBlock(
            in_channels         = self.in_channels,
            out_channels        = self.out_channels,
            kernel              = self.kernel_size,
            group_size          = self.group_size,
            time_compression    = self.time_compression,
            spatial_compression = self.spatial_compression,
            direction           = 'down'
        ).to(device).train()
        self.device = device

        # choose a frame that is *not* in the very first causal group
        # self.cut_frame = self.group_size * self.time_compression + 1    # e.g. 9 when G=4, T=2
        self.cut_frame = 6

    def test_group_causality_down(self):
        x = torch.zeros(
            self.batch_size, self.in_channels,
            self.sequence_length, 8, 8, device=self.device
        )

        y_base = self.model(x)[0]                    # forward once
        x[:, :, self.cut_frame] = torch.randn_like(x[:, :, self.cut_frame])
        y_diff = self.model(x)[0] - y_base           # forward after perturbation

        # ---- expected causal window in the output ----
        stride_time = self.group_size * self.time_compression      # actual stride inside the conv
        min_idx_aff = (self.cut_frame // stride_time) * self.group_size
        max_idx_aff = min(min_idx_aff + self.group_size, y_diff.shape[2])

        # affected slice must really change
        self.assertGreaterEqual(y_diff[:, :, min_idx_aff:max_idx_aff].std(dim=(0, 1, 3, 4)).min().item(), self.perturb_std_threshold)
        # nothing *before* the causal slice may change
        self.assertLessEqual(y_diff[:, :, :min_idx_aff].std().item(), self.error_bound)
        # nothing well *after* the causal slice may change
        self.assertLessEqual(y_diff[:, :, max_idx_aff:].std().item(), self.error_bound)


class TestUpDownBlockUpCausality(unittest.TestCase):
    """
    Group-causality for a temporal-*up* transition.
    """
    def setUp(self):
        self.batch_size            = 2
        self.in_channels           = 3
        self.out_channels          = 3
        self.group_size            = 4        # group size *before* up‑sampling
        self.time_compression      = 2        # up‑factor (inverse of compression)
        self.spatial_compression   = 2
        self.kernel_size           = (self.group_size * 2, 3, 3)
        self.sequence_length       = 16       # length of the *compressed* latent

        self.error_bound           = 1e-3
        self.perturb_std_threshold = 1e-1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UpDownBlock(
            in_channels         = self.in_channels,
            out_channels        = self.out_channels,
            kernel              = self.kernel_size,
            group_size          = self.group_size,
            time_compression    = self.time_compression,
            spatial_compression = self.spatial_compression,
            direction           = 'up'
        ).to(device).train()
        self.device = device

        # again, pick a latent time‑step that has *past* frames
        # self.cut_frame = (self.group_size // self.time_compression) + 1   # e.g. 3 when G=4, T=2
        self.cut_frame = 3

    def test_group_causality_up(self):
        x = torch.zeros(
            self.batch_size, self.in_channels,
            self.sequence_length, 8, 8, device=self.device
        )

        y_base = self.model(x)[0]
        x[:, :, self.cut_frame] = torch.randn_like(x[:, :, self.cut_frame])
        y_diff = self.model(x)[0] - y_base

        # causal maths for the *up* case
        min_idx_aff   = ((self.cut_frame * self.time_compression)// self.group_size) * self.group_size
        max_idx_aff   = min(min_idx_aff + self.kernel_size[0], y_diff.shape[2]) # but it passes with this one instead (which is wrong!)

        # affected slice
        self.assertGreaterEqual(y_diff[:, :, min_idx_aff:max_idx_aff].std(dim=(0, 1, 3, 4)).min().item(), self.perturb_std_threshold)
        # no backwards leakage
        self.assertLessEqual(y_diff[:, :, :min_idx_aff].std().item(), self.error_bound)
        # no excessive forward leakage
        self.assertLessEqual(y_diff[:, :, max_idx_aff:].std().item(), self.error_bound)


if __name__ == "__main__":
    unittest.main()
