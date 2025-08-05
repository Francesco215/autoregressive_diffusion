import torch
import einops
from ultralytics import FastSAM
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load test image
test_img_path = 'test_image.jpg'
try:
   pil_image = Image.open(test_img_path).convert('RGB')
   # Convert PIL to tensor and resize to 256x256
   test_image_np = np.array(pil_image)
   test_image = torch.from_numpy(test_image_np).float() / 255.0  # Convert to [0,1]
   test_image = test_image.permute(2, 0, 1)  # [C, H, W]
   
   # Resize to 256x256 if needed
   import torch.nn.functional as F
   if test_image.shape[1] != 256 or test_image.shape[2] != 256:
       test_image = F.interpolate(test_image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
       test_image = test_image.squeeze(0)
   
   # Create batch of 1 image
   test_images = test_image.unsqueeze(0)  # [1, C, H, W]
   batch_size = 1
   
   print(f"Loaded image shape: {test_images.shape}")
   print(f"Image range: [{test_images.min():.3f}, {test_images.max():.3f}]")
   
except FileNotFoundError:
   print(f"Could not find {test_img_path}, using random noise instead")
   batch_size = 1
   channels = 3
   height, width = 256, 256
   test_images = torch.rand(batch_size, channels, height, width)

channels = 3
height, width = 256, 256

# Initialize FastSAM
fastsam = FastSAM("FastSAM-s.pt")

# Test different confidence levels
confidence_levels = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

results_data = []

for conf in confidence_levels:
   print(f"\n=== Testing confidence {conf} ===")
   
   results = fastsam(test_images, device="cuda", retina_masks=True, imgsz=1024, conf=conf, iou=0.9, verbose=False)
   
   frame_stats = []
   for i, result in enumerate(results):
       if result.masks is not None:
           masks = result.masks.data  # Keep on GPU
           num_segments = masks.shape[0]
           
           # Calculate coverage - keep everything on GPU
           total_coverage = torch.zeros(masks.shape[1], masks.shape[2], device=masks.device)
           for mask in masks:
               total_coverage += mask.float()
           
           coverage_ratio = (total_coverage > 0).float().mean().item()
           uncovered_pixels = ((total_coverage == 0).sum()).item()
           
           frame_stats.append({
               'frame': i,
               'num_segments': num_segments,
               'coverage': coverage_ratio,
               'uncovered_pixels': uncovered_pixels
           })
           
           print(f"  Frame {i}: {num_segments} segments, {coverage_ratio:.3f} coverage, {uncovered_pixels} uncovered pixels")
       else:
           print(f"  Frame {i}: No segments found")
           frame_stats.append({
               'frame': i,
               'num_segments': 0,
               'coverage': 0.0,
               'uncovered_pixels': height * width
           })
   
   results_data.append({
       'confidence': conf,
       'results': results,
       'stats': frame_stats
   })

# Plot coverage vs confidence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Coverage plot
coverages = [data['stats'][0]['coverage'] for data in results_data]
ax1.plot(confidence_levels, coverages, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Confidence Threshold')
ax1.set_ylabel('Coverage Ratio')
ax1.set_title('Coverage vs Confidence')
ax1.grid(True)
ax1.set_ylim(0, 1.1)

# Number of segments plot
num_segments = [data['stats'][0]['num_segments'] for data in results_data]
ax2.plot(confidence_levels, num_segments, marker='o', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('Confidence Threshold')
ax2.set_ylabel('Number of Segments')
ax2.set_title('Number of Segments vs Confidence')
ax2.grid(True)

plt.tight_layout()
plt.savefig('fastsam_coverage_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualize masks for the image at different confidence levels
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Original image
axes[0].imshow(test_images[0].permute(1, 2, 0))
axes[0].set_title('Original')
axes[0].axis('off')

# Show masks for different confidence levels
viz_confs = [0.4, 0.3, 0.2, 0.1, 0.05]
for i, conf in enumerate(viz_confs):
   if i + 1 >= len(axes):
       break
       
   # Find results for this confidence
   conf_data = next((data for data in results_data if data['confidence'] == conf), None)
   if conf_data and conf_data['results'][0].masks is not None:
       masks = conf_data['results'][0].masks.data.cpu()  # Move to CPU for visualization
       
       # Create colored mask overlay
       colored_mask = torch.zeros(height, width, 3)
       colors = plt.cm.Set3(np.linspace(0, 1, min(masks.shape[0], 12)))[:, :3]
       
       for seg_idx, mask in enumerate(masks):
           color = colors[seg_idx % len(colors)]
           for c in range(3):
               colored_mask[:, :, c] += mask.float() * color[c]
       
       # Show original + overlay
       axes[i + 1].imshow(test_images[0].permute(1, 2, 0))
       axes[i + 1].imshow(colored_mask, alpha=0.5)
       
       coverage = conf_data['stats'][0]['coverage']
       num_segs = conf_data['stats'][0]['num_segments']
       axes[i + 1].set_title(f'Conf={conf}\n{num_segs} segs, {coverage:.2f} cov')
   else:
       axes[i + 1].text(0.5, 0.5, 'No segments', ha='center', va='center', transform=axes[i + 1].transAxes)
       axes[i + 1].set_title(f'Conf={conf}\nNo segments')
   
   axes[i + 1].axis('off')

# Hide unused subplots
for i in range(len(viz_confs) + 1, len(axes)):
   axes[i].axis('off')

plt.tight_layout()
plt.savefig('fastsam_masks_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plots: fastsam_coverage_analysis.png and fastsam_masks_visualization.png")