import torch
import unittest
import numpy as np

from edm2.vae import GroupCausal3DConvVAE
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
if __name__ == "__main__":
    unittest.main()
