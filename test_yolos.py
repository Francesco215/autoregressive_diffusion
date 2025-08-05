import torch
import einops
from ultralytics import YOLO
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
   
   print(f"Loaded image, resized to training size: {test_image.shape}")
   
   # Create batch of 4 images for testing
   test_images = test_image.unsqueeze(0).repeat(4, 1, 1, 1)  # [4, C, H, W]
   
except FileNotFoundError:
   print(f"Could not find {test_img_path}, creating random 256x256 images")
   channels = 3
   test_images = torch.rand(4, channels, original_size, original_size)

print(f"Test images shape: {test_images.shape}")
print(f"Image range: [{test_images.min():.3f}, {test_images.max():.3f}]")

# Test ALL YOLO models and compare them
yolo_models = [
    "yolo11n-seg.pt",  # Newest, smallest
    "yolov8n-seg.pt",  # YOLOv8 nano segmentation
    "yolov8s-seg.pt",  # YOLOv8 small segmentation
    "yolo11s-seg.pt",  # YOLO11 small segmentation
]

all_models = {}
for model_name in yolo_models:
    try:
        print(f"\n=== Loading {model_name} ===")
        yolo = YOLO(model_name)
        print(f"Successfully loaded {model_name}")
        all_models[model_name] = yolo
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")

if not all_models:
    print("Could not load any YOLO segmentation models!")
    exit()

print(f"\nLoaded {len(all_models)} models: {list(all_models.keys())}")

# Test each model
comparison_results = {}

for model_name, model in all_models.items():
    print(f"\n{'='*20} Testing {model_name} {'='*20}")
    
    # Test with first image
    test_img = test_images[0:1]
    
    try:
        results = model(test_img, device="cuda", verbose=False)
        result = results[0]
        
        if result.masks is not None:
            masks = result.masks.data
            num_segments = masks.shape[0]
            
            # Get classes
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[cls] for cls in classes]
            else:
                class_names = []
            
            # Calculate coverage
            total_coverage = torch.zeros(masks.shape[1], masks.shape[2], device=masks.device)
            for mask in masks:
                total_coverage += mask.float()
            coverage_ratio = (total_coverage > 0).float().mean().item()
            
            comparison_results[model_name] = {
                'segments': num_segments,
                'classes': class_names,
                'coverage': coverage_ratio,
                'masks': masks.cpu(),
                'success': True
            }
            
            print(f"✓ {num_segments} segments, {coverage_ratio:.3f} coverage")
            print(f"  Classes: {class_names}")
            
        else:
            comparison_results[model_name] = {
                'segments': 0,
                'classes': [],
                'coverage': 0.0,
                'masks': None,
                'success': False
            }
            print("✗ No segments found")
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        comparison_results[model_name] = {
            'segments': 0,
            'classes': [],
            'coverage': 0.0,
            'masks': None,
            'success': False
        }

# Visualize comparison
print(f"\n=== Model Comparison Visualization ===")
successful_models = {k: v for k, v in comparison_results.items() if v['success']}

if successful_models:
    num_models = len(successful_models)
    fig, axes = plt.subplots(2, max(2, (num_models + 1) // 2), figsize=(16, 8))
    if num_models == 1:
        axes = axes.reshape(2, 1)
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(test_images[0].permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Each model result
    for idx, (model_name, data) in enumerate(successful_models.items()):
        ax_idx = idx + 1
        if ax_idx >= len(axes):
            break
            
        masks = data['masks']
        if masks is not None:
            # Create colored overlay
            colored_mask = torch.zeros(masks.shape[1], masks.shape[2], 3)
            colors = plt.cm.Set3(np.linspace(0, 1, min(masks.shape[0], 12)))[:, :3]
            
            for seg_idx, mask in enumerate(masks):
                color = colors[seg_idx % len(colors)]
                for c in range(3):
                    colored_mask[:, :, c] += mask.float() * color[c]
            
            axes[ax_idx].imshow(test_images[0].permute(1, 2, 0))
            axes[ax_idx].imshow(colored_mask, alpha=0.6)
            axes[ax_idx].set_title(f'{model_name}\n{data["segments"]} segs, {data["coverage"]:.2f} cov')
        else:
            axes[ax_idx].text(0.5, 0.5, 'No segments', ha='center', va='center', transform=axes[ax_idx].transAxes)
            axes[ax_idx].set_title(f'{model_name}\nNo segments')
        
        axes[ax_idx].axis('off')
    
    # Hide unused subplots
    for i in range(len(successful_models) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('yolo_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved comparison: yolo_model_comparison.png")

# Print summary table
print(f"\n=== Summary Table ===")
print(f"{'Model':<15} {'Segments':<8} {'Coverage':<8} {'Classes'}")
print("-" * 60)
for model_name, data in comparison_results.items():
    if data['success']:
        classes_str = ', '.join(data['classes'][:3])  # Show first 3 classes
        if len(data['classes']) > 3:
            classes_str += f" (+{len(data['classes'])-3} more)"
        print(f"{model_name:<15} {data['segments']:<8} {data['coverage']:<8.3f} {classes_str}")
    else:
        print(f"{model_name:<15} {'FAILED':<8} {'N/A':<8} {'N/A'}")