import torch
import einops

#from edm2.vae.vae import UpDownBlock
import torch
import torch.nn as nn
import einops

class UpDownBlock:
    def __init__(self, time_compression, spatial_compression, direction):
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'
        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression
        self.total_compression = time_compression*spatial_compression**2

    def __call__(self, x):
        if self.total_compression==1: return x

        if self.direction=='down':
            return einops.rearrange(x, 'b c (t tc) (h hc) (w wc) -> b (c tc hc wc) t h w', tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)

        return einops.rearrange(x, 'b (c tc hc wc) t h w -> b c (t tc) (h hc) (w wc)', tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)

# Test basic up/down block
batch_size = 2
channels = 3
time_dim = 4
height = 8
width = 8
time_compression = 2
spatial_compression = 2

# Make a random tensor for testing
x_down_input = torch.randn(batch_size, channels, time_dim, height, width)
print("Down input shape:", x_down_input.shape)

# Try downsampling
down_block = UpDownBlock(time_compression, spatial_compression, 'down')
x_down_output = down_block(x_down_input)
print("Down output shape:", x_down_output.shape)

# Expected shape after downsampling
expected_channels = channels * time_compression * spatial_compression * spatial_compression
expected_time = time_dim // time_compression
expected_height = height // spatial_compression
expected_width = width // spatial_compression
print("Expected shape:", (batch_size, expected_channels, expected_time, expected_height, expected_width))

# Try upsampling with the downsampled output
up_block = UpDownBlock(time_compression, spatial_compression, 'up')
x_up_output = up_block(x_down_output)
print("Up output shape:", x_up_output.shape)

# Check if it worked correctly
if x_down_input.shape == x_up_output.shape:
    print("Success! Original input and up output shapes match")
else:
    print("Error: shapes don't match")

# Also test the case with no compression
no_comp_block = UpDownBlock(1, 1, 'down')
x_no_comp_output = no_comp_block(x_down_input)
print("No compression output (should be same as input):", x_no_comp_output.shape)

# Print the actual content difference to see if values are preserved reasonably
print("Max difference between original and reconstructed:", torch.max(torch.abs(x_down_input - x_up_output)).item())


#%%
import torch
import numpy as np
from edm2.vae.vae import GroupCausal3DConvVAE
import torch.nn.functional as F

class UpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, time_compression, spatial_compression, direction):
        super().__init__()
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'

        self.in_channels, self.out_channels, self.kernel, self.group_size = in_channels, out_channels, kernel, group_size
        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression
        self.total_compression = time_compression*spatial_compression**2

        kernel = (kernel[0]//time_compression, kernel[1], kernel[2])
        if direction=='up':
            group_size = group_size//time_compression
            self.stride = [1,1,1]
        if direction=='down':
            self.stride = [time_compression, spatial_compression, spatial_compression]

        self.conv = GroupCausal3DConvVAE(in_channels, out_channels, kernel, group_size, stride = self.stride) 

    def __call__(self, x, cache=None):
        x, cache = self.conv(x, cache)

        if self.direction=='up' and self.total_compression !=1:
            x = F.interpolate(x, scale_factor=[self.time_compression, self.spatial_compression, self.spatial_compression], mode='nearest')

        return x, cache
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, time_compression={self.time_compression}, space_compression={self.spatial_compression}, kernel={self.kernel}, group_size={self.group_size}, stride={self.stride})"
    
    def _load_from_2D_module(self, UpDownSampler):
        self.conv._load_from_2D_module(UpDownSampler.conv)
        
        
# Test for UpDownBlock - testing both downsampling and upsampling operations

# Set random seed for reproducibility
torch.manual_seed(42)

# Test parameters
in_channels = 4
out_channels = 8
kernel = (8, 3, 3)
group_size = 4
time_compression = 2
spatial_compression = 2
batch_size = 2
time_dim = 16
height = 32
width = 32

# Create a sample input tensor (batch, channels, time, height, width)
x = torch.randn(batch_size, in_channels, time_dim, height, width)

print(f"Input shape: {x.shape}")

# ----- Test downsampling -----
print("\n----- Testing Downsampling -----")

# Create downsampling block
down_block = UpDownBlock(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel=kernel,
    group_size=group_size,
    time_compression=time_compression,
    spatial_compression=spatial_compression,
    direction='down'
)

# Forward pass
output_down, cache_down = down_block(x)

# Expected output dimensions after downsampling
expected_time = time_dim // time_compression
expected_height = height // spatial_compression
expected_width = width // spatial_compression
expected_shape = (batch_size, out_channels, expected_time, expected_height, expected_width)

print(f"Downsampled output shape: {output_down.shape}")
print(f"Expected shape: {expected_shape}")
print(f"Shapes match: {output_down.shape == expected_shape}")

# ----- Test upsampling -----
print("\n----- Testing Upsampling -----")

# Create upsampling block
up_block = UpDownBlock(
    in_channels=out_channels,
    out_channels=in_channels,
    kernel=(kernel[0]//time_compression, kernel[1], kernel[2]),  # Adjust kernel for upsampling
    group_size=group_size//time_compression,  # Adjust group size for upsampling
    time_compression=time_compression,
    spatial_compression=spatial_compression,
    direction='up'
)

# Forward pass with downsampled output
output_up, cache_up = up_block(output_down)

# Expected output dimensions after upsampling back to original size
expected_up_shape = (batch_size, in_channels, time_dim, height, width)

print(f"Upsampled output shape: {output_up.shape}")
print(f"Expected shape: {expected_up_shape}")
print(f"Shapes match: {output_up.shape == expected_up_shape}")

# ----- Test with cache -----
print("\n----- Testing with Cache -----")

# Create a new input
x_new = torch.randn(batch_size, in_channels, time_dim, height, width)

# Forward pass with cache
output_down_cache, new_cache = down_block(x_new, cache_down)

print(f"Output with cache shape: {output_down_cache.shape}")
print(f"Cache is not None: {new_cache is not None}")

# ----- Test values are different -----
print("\n----- Testing Output Values -----")

# Check that different inputs produce different outputs
are_outputs_different = not torch.allclose(output_down, output_down_cache)
print(f"Different inputs produce different outputs: {are_outputs_different}")

print("\nAll tests completed!")

#%%

class UpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, 
                 time_compression, spatial_compression, direction):
        super().__init__()
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_orig = kernel 
        self.group_size_orig = group_size 
        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression

        # Effective channels and stride for the convolutional layer
        conv_in_channels = in_channels
        conv_out_channels = out_channels
        conv_stride = [1, 1, 1] # temporal stride is always 1 for conv
                                # because einops handles temporal compression

        if direction == 'down':

            conv_in_channels = in_channels * time_compression
            conv_stride = [1, spatial_compression, spatial_compression]
            conv_kernel_config = kernel 
            conv_group_config = group_size

        elif direction == 'up':
            conv_out_channels = out_channels * time_compression
            conv_stride = [1, 1, 1]
            conv_kernel_config = kernel
            conv_group_config = group_size
        
        self.conv_stride_config = conv_stride # For __repr__

        self.conv = GroupCausal3DConvVAE(
            conv_in_channels, 
            conv_out_channels, 
            conv_kernel_config, 
            conv_group_config, 
            stride=conv_stride
        )

    def forward(self, x, cache=None):
        # x shape: (b, c, t, h, w)

        if self.direction == 'down':
            if self.time_compression > 1:
                # (b, c, t*tc, h, w) -> (b, c*tc, t, h, w)
                x = einops.rearrange(x, 'b c (t tc) h w -> b (c tc) t h w', tc=self.time_compression)
            # Input to conv: (b, in_c*tc, t, h, w)
            # Output of conv: (b, out_c, t, h/sc, w/sc) if conv changes channels from in_c*tc to out_c
            x, cache = self.conv(x, cache)

        elif self.direction == 'up':
            # Input to conv: (b, in_c, t, h, w)
            # Output of conv: (b, out_c*tc, t, h, w)
            x, cache = self.conv(x, cache)
            if self.spatial_compression > 1:
                # (b, out_c*tc, t, h, w) -> (b, out_c*tc, t, h*sc, w*sc)
                x = F.interpolate(x, scale_factor=[1, self.spatial_compression, self.spatial_compression], mode='nearest')
            if self.time_compression > 1:
                # (b, out_c*tc, t, h*sc, w*sc) -> (b, out_c, t*tc, h*sc, w*sc)
                x = einops.rearrange(x, 'b (c tc) t h w -> b c (t tc) h w', tc=self.time_compression)
        
        return x, cache
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"time_compression={self.time_compression}, spatial_compression={self.spatial_compression}, "
                f"kernel={self.kernel_orig}, group_size={self.group_size_orig}, direction='{self.direction}', "
                f"conv_stride={self.conv_stride_config})") # Show actual conv stride
    
    def _load_from_2D_module(self, UpDownSampler_2D):
        if hasattr(UpDownSampler_2D, 'conv'):
            self.conv._load_from_2D_module(UpDownSampler_2D.conv)
        else:
            print(f"Warning: UpDownSampler_2D of type {type(UpDownSampler_2D)} does not have a 'conv' attribute.")
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

# Import your implementation - adjust path as needed

# Test function for the combined UpDownBlock
print("STARTING TEST FOR COMBINED UPDOWN BLOCK")
print("=======================================")

# Step 1: Set up test parameters
print("\nStep 1: Setting up test parameters")
batch_size = 2
in_channels = 4
out_channels = 8
time_dim = 16
height = 32
width = 32
kernel = (8, 3, 3)
group_size = 4
time_compression = 2
spatial_compression = 2

print(f"Test parameters:")
print(f"- Batch size: {batch_size}")
print(f"- Input channels: {in_channels}, Output channels: {out_channels}")
print(f"- Input dimensions: Time={time_dim}, Height={height}, Width={width}")
print(f"- Kernel size: {kernel}")
print(f"- Group size: {group_size}")
print(f"- Compression factors: Time={time_compression}, Spatial={spatial_compression}")

# Step 2: Create input tensor
print("\nStep 2: Creating input tensor")
torch.manual_seed(42)  # For reproducibility
x = torch.randn(batch_size, in_channels, time_dim, height, width)
print(f"Input tensor shape: {x.shape}")

# Step 3: Create and test downsampling block
print("\nStep 3: Testing downsampling")
down_block = UpDownBlock(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel=kernel,
    group_size=group_size,
    time_compression=time_compression,
    spatial_compression=spatial_compression,
    direction='down'
)
print(f"Created downsampling block: {down_block}")

# Run forward pass
down_output, down_cache = down_block(x)

# Check results
expected_down_shape = (batch_size, out_channels, time_dim // time_compression, 
                        height // spatial_compression, width // spatial_compression)
print(f"Downsampled output shape: {down_output.shape}")
print(f"Expected shape: {expected_down_shape}")

is_shape_correct = down_output.shape == expected_down_shape
print(f"Shape matches expected: {'✓ YES' if is_shape_correct else '✗ NO'}")

# Step 4: Create and test upsampling block
print("\nStep 4: Testing upsampling")
up_block = UpDownBlock(
    in_channels=out_channels,
    out_channels=in_channels,
    kernel=(kernel[0]//time_compression, kernel[1], kernel[2]),
    group_size=group_size//time_compression,
    time_compression=time_compression,
    spatial_compression=spatial_compression,
    direction='up'
)
print(f"Created upsampling block: {up_block}")

# Run forward pass
up_output, up_cache = up_block(down_output)

# Check results
expected_up_shape = (batch_size, in_channels, time_dim, height, width)
print(f"Upsampled output shape: {up_output.shape}")
print(f"Expected shape: {expected_up_shape}")

is_shape_correct = up_output.shape == expected_up_shape
print(f"Shape matches expected: {'✓ YES' if is_shape_correct else '✗ NO'}")

# Step 5: Test round trip content preservation
print("\nStep 5: Testing content preservation")
# We don't expect exact matching due to lossy compression/decompression,
# but the outputs shouldn't be completely random either

# Calculate mean absolute difference
mean_abs_diff = torch.abs(x - up_output).mean().item()
print(f"Mean absolute difference between input and reconstructed output: {mean_abs_diff:.6f}")

# Check correlation
x_flat = x.flatten()
up_flat = up_output.flatten()
correlation = torch.corrcoef(torch.stack([x_flat, up_flat]))[0, 1].item()
print(f"Correlation between input and reconstructed output: {correlation:.6f}")

# Step 6: Test with eval mode (cache functionality)
print("\nStep 6: Testing cache functionality in eval mode")
down_block.eval()  # Set to evaluation mode

# Run with initial input and get cache
initial_output, initial_cache = down_block(x)
print(f"Cache is None: {'✓ YES' if initial_cache is None else '✗ NO (expected in training mode)'}")

# Create a new sequential input
x_next = torch.randn(batch_size, in_channels, time_dim, height, width)

# Run with new input and previous cache
next_output, next_cache = down_block(x_next, initial_cache)
print(f"Output with cache shape: {next_output.shape}")

print("\nTest completed!")
#%%
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

# First implementation - simplified version
class UpDownBlockOriginal:
    def __init__(self, time_compression, spatial_compression, direction):
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'
        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression
        self.total_compression = time_compression*spatial_compression**2

    def __call__(self, x):
        if self.total_compression==1: return x

        if self.direction=='down':
            return einops.rearrange(x, 'b c (t tc) (h hc) (w wc) -> b (c tc hc wc) t h w', 
                                   tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)

        return einops.rearrange(x, 'b (c tc hc wc) t h w -> b c (t tc) (h hc) (w wc)', 
                               tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)


# Final implementation
class UpDownBlockFinal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, 
                 time_compression, spatial_compression, direction):
        super().__init__()
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_orig = kernel 
        self.group_size_orig = group_size 
        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression

        # Effective channels and stride for the convolutional layer
        conv_in_channels = in_channels
        conv_out_channels = out_channels
        conv_stride = [1, 1, 1]

        if direction == 'down':
            conv_in_channels = in_channels * time_compression
            conv_stride = [1, spatial_compression, spatial_compression]
            conv_kernel_config = kernel 
            conv_group_config = group_size

        elif direction == 'up':
            conv_out_channels = out_channels * time_compression
            conv_stride = [1, 1, 1]
            conv_kernel_config = kernel
            conv_group_config = group_size
        
        self.conv_stride_config = conv_stride

        self.conv = GroupCausal3DConvVAE(
            conv_in_channels, 
            conv_out_channels, 
            conv_kernel_config, 
            conv_group_config, 
            stride=conv_stride
        )

    def forward(self, x, cache=None):
        # x shape: (b, c, t, h, w)

        if self.direction == 'down':
            if self.time_compression > 1:
                # (b, c, t*tc, h, w) -> (b, c*tc, t, h, w)
                x = einops.rearrange(x, 'b c (t tc) h w -> b (c tc) t h w', tc=self.time_compression)
            # Input to conv: (b, in_c*tc, t, h, w)
            # Output of conv: (b, out_c, t, h/sc, w/sc) if conv changes channels from in_c*tc to out_c
            x, cache = self.conv(x, cache)

        elif self.direction == 'up':
            # Input to conv: (b, in_c, t, h, w)
            # Output of conv: (b, out_c*tc, t, h, w)
            x, cache = self.conv(x, cache)
            if self.spatial_compression > 1:
                # (b, out_c*tc, t, h, w) -> (b, out_c*tc, t, h*sc, w*sc)
                x = F.interpolate(x, scale_factor=[1, self.spatial_compression, self.spatial_compression], mode='nearest')
            if self.time_compression > 1:
                # (b, out_c*tc, t, h*sc, w*sc) -> (b, out_c, t*tc, h*sc, w*sc)
                x = einops.rearrange(x, 'b (c tc) t h w -> b c (t tc) h w', tc=self.time_compression)
        
        return x, cache

# Testing equivalence between original and final implementations
def test_equivalence():
    print("TESTING EQUIVALENCE BETWEEN ORIGINAL AND FINAL UPDOWNBLOCK IMPLEMENTATIONS")
    print("======================================================================")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    in_channels = 3
    out_channels = 6
    time_dim = 8
    height = 16
    width = 16
    time_compression = 2
    spatial_compression = 2
    kernel = (4, 3, 3)
    group_size = 2
    
    print(f"\nTest Parameters:")
    print(f"- Input shape: [batch={batch_size}, channels={in_channels}, time={time_dim}, height={height}, width={width}]")
    print(f"- Compression: time={time_compression}, spatial={spatial_compression}")
    
    # Create test tensors
    x_orig = torch.randn(batch_size, in_channels, time_dim, height, width)
    x_final = x_orig.clone()  # Use the same input for both implementations
    
    print("\n1. Testing DOWNSAMPLING operation:")
    
    # Original implementation (simplified rearrangement only)
    orig_down = UpDownBlockOriginal(time_compression, spatial_compression, 'down')
    output_orig_down = orig_down(x_orig)
    
    # Final implementation (with convolution)
    final_down = UpDownBlockFinal(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel=kernel,
        group_size=group_size,
        time_compression=time_compression,
        spatial_compression=spatial_compression,
        direction='down'
    )
    output_final_down, _ = final_down(x_final)
    
    # Compare shapes after downsampling
    print(f"Original implementation output shape: {output_orig_down.shape}")
    print(f"Final implementation output shape: {output_final_down.shape}")
    
    # Expected shape for original implementation
    expected_orig_channels = in_channels * time_compression * spatial_compression * spatial_compression
    expected_orig_shape = (batch_size, expected_orig_channels, 
                           time_dim // time_compression, 
                           height // spatial_compression, 
                           width // spatial_compression)
    
    # Expected shape for final implementation
    expected_final_shape = (batch_size, out_channels, 
                           time_dim // time_compression, 
                           height // spatial_compression, 
                           width // spatial_compression)
    
    print(f"Expected original shape: {expected_orig_shape}")
    print(f"Expected final shape: {expected_final_shape}")
    
    shape_match_orig = output_orig_down.shape == expected_orig_shape
    shape_match_final = output_final_down.shape == expected_final_shape
    
    print(f"Original matches expected shape: {'✓' if shape_match_orig else '✗'}")
    print(f"Final matches expected shape: {'✓' if shape_match_final else '✗'}")
    
    # Key differences analysis
    print("\nKey differences analysis for downsampling:")
    print("1. Original: pure reshaping without learning parameters")
    print("2. Final: includes convolution layer that can transform channels")
    print("3. Channel dimension changes: original uses fixed channel expansion, final uses learnable transformation")
    
    print("\n2. Testing UPSAMPLING operation:")
    
    # For upsampling test, we need a tensor with appropriate shape
    # Original implementation
    x_up_orig = output_orig_down.clone()
    orig_up = UpDownBlockOriginal(time_compression, spatial_compression, 'up')
    output_orig_up = orig_up(x_up_orig)
    
    # Final implementation
    # Note: For accurate comparison, we'd need to use the right channel dimension
    # Since original and final have different channel counts after downsampling,
    # we'll create a separate test tensor for final upsampling
    down_final_out_ch = output_final_down.shape[1]
    final_up = UpDownBlockFinal(
        in_channels=down_final_out_ch,
        out_channels=in_channels,
        kernel=(kernel[0]//time_compression, kernel[1], kernel[2]),
        group_size=group_size//time_compression,
        time_compression=time_compression,
        spatial_compression=spatial_compression,
        direction='up'
    )
    output_final_up, _ = final_up(output_final_down)
    
    # Compare shapes after upsampling
    print(f"Original implementation output shape: {output_orig_up.shape}")
    print(f"Final implementation output shape: {output_final_up.shape}")
    
    # Expected shape after upsampling (should match original input)
    expected_up_shape = (batch_size, in_channels, time_dim, height, width)
    
    shape_match_orig_up = output_orig_up.shape == expected_up_shape
    shape_match_final_up = output_final_up.shape == expected_up_shape
    
    print(f"Original up matches input shape: {'✓' if shape_match_orig_up else '✗'}")
    print(f"Final up matches input shape: {'✓' if shape_match_final_up else '✗'}")
    
    # End-to-end test
    print("\n3. Testing ROUND-TRIP (down then up):")
    print(f"Original input shape: {x_orig.shape}")
    print(f"Original round-trip output shape: {output_orig_up.shape}")
    print(f"Final round-trip output shape: {output_final_up.shape}")
    
    # Check if original input and up output match in shape
    round_trip_shape_match_orig = x_orig.shape == output_orig_up.shape
    round_trip_shape_match_final = x_orig.shape == output_final_up.shape
    
    print(f"Original round-trip shape preserved: {'✓' if round_trip_shape_match_orig else '✗'}")
    print(f"Final round-trip shape preserved: {'✓' if round_trip_shape_match_final else '✗'}")
    
    # Key differences analysis for the implementations
    print("\nOverall comparison between implementations:")
    print("1. Original implementation:")
    print("   - Pure reshaping operations using einops")
    print("   - No learnable parameters")
    print("   - Fixed channel transformation based on compression factors")
    
    print("\n2. Final implementation:")
    print("   - Uses convolutional layers with learnable parameters")
    print("   - Handles time and spatial dimensions differently")
    print("   - Can change channel dimensions independently of compression factors")
    print("   - Supports caching for inference/evaluation")
    print("   - More flexible but computationally more expensive")
    
    print("\nConclusion:")
    if shape_match_orig and shape_match_final and shape_match_orig_up and shape_match_final_up:
        print("Both implementations produce outputs with the expected shapes")
        print("However, they are functionally DIFFERENT in their approach and capabilities:")
        print("- Original is a pure reshaping operation")
        print("- Final adds convolution for feature transformation")
    else:
        print("There are shape mismatches between implementations")
    
    return {
        "original_down": output_orig_down,
        "final_down": output_final_down,
        "original_up": output_orig_up,
        "final_up": output_final_up
    }

# Run the test
results = test_equivalence()