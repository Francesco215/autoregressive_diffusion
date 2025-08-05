import torch
import einops
from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

# Create test data at 256x256 (our training size)
test_img_path = 'test_image.jpg'
original_size = 256

try:
   pil_image = Image.open(test_img_path).convert('RGB')
   # Convert PIL to tensor and resize to our training size (256x256)
   test_image_np = np.array(pil_image)
   test_image = torch.from_numpy(test_image_np).float() / 255.0  # Convert to [0,1]
   test_image = test_image.permute(2, 0, 1)  # [C, H, W]
   
   # Resize to 256x256 (our training resolution)
   if test_image.shape[1] != original_size or test_image.shape[2] != original_size:
       test_image = F.interpolate(test_image.unsqueeze(0), size=(original_size, original_size), mode='bilinear', align_corners=False)
       test_image = test_image.squeeze(0)
   
   print(f"Loaded test_image.jpg, resized to training size: {test_image.shape}")
   
   # Create batch of 4 images for testing
   test_images_256 = test_image.unsqueeze(0).repeat(4, 1, 1, 1)  # [4, C, H, W]
   
except FileNotFoundError:
   print(f"Could not find {test_img_path}, creating random 256x256 images")
   channels = 3
   test_images_256 = torch.rand(4, channels, original_size, original_size)

print(f"Test images shape (training size): {test_images_256.shape}")
print(f"Image range: [{test_images_256.min():.3f}, {test_images_256.max():.3f}]")

# Try different SAM models from smallest to largest
sam_models = [
   "mobile_sam.pt",    # Smallest
   "sam2_t.pt",        # SAM2 tiny
   "sam2_b.pt",        # SAM2 base  
   "sam_b.pt",         # Original SAM base
]

# Test all SAM models
all_models = {}
for model_name in sam_models:
   try:
       print(f"\n=== Loading {model_name} ===")
       sam = SAM(model_name)
       print(f"Successfully loaded {model_name}")
       all_models[model_name] = sam
   except Exception as e:
       print(f"Failed to load {model_name}: {e}")

if not all_models:
   print("Could not load any SAM models!")
   exit()

print(f"\nLoaded {len(all_models)} SAM models: {list(all_models.keys())}")

# Function to upscale -> segment -> downscale
def segment_with_upscale_downscale(image_256, sam_model, model_name):
   """
   Takes 256x256 image, upscales to 1024x1024 for SAM, then downscales masks back to 256x256
   """
   try:
       # Upscale to 1024x1024 for SAM
       image_1024 = F.interpolate(image_256, size=(1024, 1024), mode='bilinear', align_corners=False)
       
       # Run SAM segmentation
       with torch.no_grad():
           result = sam_model(image_1024, device="cuda", verbose=False)
       
       if len(result) == 0 or result[0].masks is None:
           return None, None, f"No masks generated"
       
       # Get masks at 1024x1024
       masks_1024 = result[0].masks.data  # [N_segments, 1024, 1024]
       
       # Downscale masks back to 256x256
       masks_256 = F.interpolate(
           masks_1024.float().unsqueeze(1),  # [N_segments, 1, 1024, 1024]
           size=(original_size, original_size), 
           mode='bilinear', 
           align_corners=False
       ).squeeze(1)  # [N_segments, 256, 256]
       
       # Threshold back to binary masks (since interpolation makes them fuzzy)
       masks_256 = (masks_256 > 0.5).float()
       
       return masks_256, result[0], "Success"
       
   except Exception as e:
       return None, None, f"Error: {str(e)}"

# Test each SAM model
comparison_results = {}

for model_name, model in all_models.items():
   print(f"\n{'='*20} Testing {model_name} {'='*20}")
   
   # Test with first image
   single_image = test_images_256[0:1]  # [1, C, 256, 256]
   masks_256, original_result, status = segment_with_upscale_downscale(single_image, model, model_name)
   
   if masks_256 is not None:
       num_segments = masks_256.shape[0]
       
       # Calculate coverage at 256x256
       total_coverage = torch.zeros(original_size, original_size)
       for mask in masks_256:
           total_coverage += mask.cpu()
       
       coverage_ratio = (total_coverage > 0).float().mean().item()
       uncovered_pixels = ((total_coverage == 0).sum()).item()
       
       comparison_results[model_name] = {
           'segments': num_segments,
           'coverage': coverage_ratio,
           'uncovered_pixels': uncovered_pixels,
           'masks': masks_256.cpu(),
           'success': True,
           'status': status
       }
       
       print(f"✓ {num_segments} segments, {coverage_ratio:.3f} coverage, {uncovered_pixels} uncovered pixels")
       
       # Test batch processing capability
       try:
           print("  Testing batch processing...")
           batch_results = []
           for i in range(min(2, test_images_256.shape[0])):  # Test 2 images
               img = test_images_256[i:i+1]
               masks, _, status = segment_with_upscale_downscale(img, model, model_name)
               if masks is not None:
                   batch_results.append(masks.shape[0])
               else:
                   batch_results.append(0)
           
           print(f"  Batch test: {batch_results} segments per image")
           comparison_results[model_name]['batch_test'] = batch_results
           
       except Exception as e:
           print(f"  Batch test failed: {e}")
           comparison_results[model_name]['batch_test'] = "Failed"
       
   else:
       comparison_results[model_name] = {
           'segments': 0,
           'coverage': 0.0,
           'uncovered_pixels': original_size * original_size,
           'masks': None,
           'success': False,
           'status': status,
           'batch_test': "N/A"
       }
       print(f"✗ {status}")

# Visualize comparison
print(f"\n=== Model Comparison Visualization ===")
successful_models = {k: v for k, v in comparison_results.items() if v['success']}

if successful_models:
   num_models = len(successful_models)
   fig, axes = plt.subplots(2, max(2, (num_models + 1) // 2), figsize=(16, 10))
   if num_models == 1:
       axes = axes.reshape(2, 1)
   axes = axes.flatten()
   
   # Original image
   axes[0].imshow(test_images_256[0].permute(1, 2, 0))
   axes[0].set_title('Original 256x256')
   axes[0].axis('off')
   
   # Each model result
   colors = plt.cm.Set3(np.linspace(0, 1, 12))[:, :3]
   
   for idx, (model_name, data) in enumerate(successful_models.items()):
       ax_idx = idx + 1
       if ax_idx >= len(axes):
           break
           
       masks = data['masks']
       if masks is not None:
           # Create colored overlay
           colored_mask = torch.zeros(original_size, original_size, 3)
           
           for seg_idx, mask in enumerate(masks):
               color = colors[seg_idx % len(colors)]
               for c in range(3):
                   colored_mask[:, :, c] += mask.float() * color[c]
           
           axes[ax_idx].imshow(test_images_256[0].permute(1, 2, 0))
           axes[ax_idx].imshow(colored_mask, alpha=0.6)
           
           short_name = model_name.replace('.pt', '').replace('_', '-')
           axes[ax_idx].set_title(f'{short_name}\n{data["segments"]} segs, {data["coverage"]:.2f} cov')
       else:
           axes[ax_idx].text(0.5, 0.5, 'No segments', ha='center', va='center', transform=axes[ax_idx].transAxes)
           axes[ax_idx].set_title(f'{model_name}\nFailed')
       
       axes[ax_idx].axis('off')
   
   # Hide unused subplots
   for i in range(len(successful_models) + 1, len(axes)):
       axes[i].axis('off')
   
   plt.tight_layout()
   plt.savefig('sam_model_comparison.png', dpi=150, bbox_inches='tight')
   plt.show()
   
   print(f"Saved comparison: sam_model_comparison.png")

# Print detailed summary table
print(f"\n=== SAM Model Comparison Summary ===")
print(f"{'Model':<15} {'Status':<8} {'Segments':<8} {'Coverage':<8} {'Uncovered':<10} {'Batch Test'}")
print("-" * 85)

for model_name, data in comparison_results.items():
   short_name = model_name.replace('.pt', '')
   if data['success']:
       batch_str = str(data['batch_test']) if isinstance(data['batch_test'], list) else data['batch_test']
       print(f"{short_name:<15} {'✓':<8} {data['segments']:<8} {data['coverage']:<8.3f} {data['uncovered_pixels']:<10} {batch_str}")
   else:
       print(f"{short_name:<15} {'✗':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A'}")

# Recommend best model
if successful_models:
   # Find model with best coverage
   best_coverage = max(successful_models.items(), key=lambda x: x[1]['coverage'])
   # Find model with most segments  
   most_segments = max(successful_models.items(), key=lambda x: x[1]['segments'])
   
   print(f"\n=== Recommendations ===")
   print(f"Best Coverage: {best_coverage[0]} ({best_coverage[1]['coverage']:.3f} coverage)")
   print(f"Most Segments: {most_segments[0]} ({most_segments[1]['segments']} segments)")
   
   if best_coverage[0] == most_segments[0]:
       print(f"🏆 Overall Best: {best_coverage[0]}")
   else:
       print(f"💡 Consider: {best_coverage[0]} for coverage, {most_segments[0]} for detail")

else:
   print("No SAM models worked successfully!")

print(f"\nMethod: Upscale 256x256 → 1024x1024 → SAM → Downscale masks → 256x256")
print("SAM model testing complete!")