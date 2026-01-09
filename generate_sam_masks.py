import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def generate_sam_masks(pseudo_label_dir, image_dir, output_dir, checkpoint_path, model_type="vit_b"):
    """
    Generate SAM masks for images that have pseudo labels
    
    Args:
        pseudo_label_dir: Path to folder containing pseudo labels (e.g., "pseudo_labels/transcam")
        image_dir: Path to folder containing original images
        output_dir: Path to save SAM masks
        checkpoint_path: Path to SAM checkpoint file
        model_type: Type of SAM model ("vit_h", "vit_l", or "vit_b")
    """
    
    # Load SAM model
    print(f"Loading SAM model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Warning: CUDA not available, using CPU (this will be very slow)")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Create mask generator with optimized parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Get all pseudo label files
    pseudo_files = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.png')]
    
    print(f"Found {len(pseudo_files)} pseudo labels in {pseudo_label_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each pseudo label
    processed_count = 0
    skipped_count = 0
    
    for pseudo_file in tqdm(pseudo_files, desc="Generating SAM masks"):
        # Extract image name (without extension)
        img_name = os.path.splitext(pseudo_file)[0]
        
        # Create output folder for this image
        output_img_dir = os.path.join(output_dir, img_name)
        
        # Skip if already processed
        if os.path.exists(output_img_dir) and len(os.listdir(output_img_dir)) > 0:
            print(f"Skipping {img_name} - already processed")
            skipped_count += 1
            continue
        
        Path(output_img_dir).mkdir(parents=True, exist_ok=True)
        
        # Find corresponding original image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            potential_path = os.path.join(image_dir, f"{img_name}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            print(f"Warning: Original image not found for {img_name}, skipping...")
            continue
        
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Failed to read image {img_path}, skipping...")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            masks = mask_generator.generate(image)
            
            # Save each mask as separate PNG file
            for idx, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                
                # Convert boolean mask to 0/255
                mask_img = (mask * 255).astype(np.uint8)
                
                # Save mask
                mask_path = os.path.join(output_img_dir, f"mask_{idx}.png")
                Image.fromarray(mask_img).save(mask_path)
            
            processed_count += 1
            print(f"Generated {len(masks)} masks for {img_name}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue
    
    print(f"\n=== Summary ===")
    print(f"Total pseudo labels: {len(pseudo_files)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Failed: {len(pseudo_files) - processed_count - skipped_count}")
    print(f"SAM masks saved to: {output_dir}")

if __name__ == "__main__":
    # Cấu hình đường dẫn
    PSEUDO_LABEL_DIR = "pseudo_labels/transcam"
    IMAGE_DIR = "../archive/VOC2012_train_val/VOC2012_train_val/JPEGImages"
    OUTPUT_DIR = "SAM_masks/voc12_transcam"
    CHECKPOINT_PATH = "pretrained_models/sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"  # Sử dụng model vit_b vì đã có checkpoint
    
    # Kiểm tra các đường dẫn
    if not os.path.exists(PSEUDO_LABEL_DIR):
        print(f"Error: Pseudo label directory not found: {PSEUDO_LABEL_DIR}")
        exit(1)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        exit(1)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: SAM checkpoint not found: {CHECKPOINT_PATH}")
        print("Please download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        exit(1)
    
    # Generate SAM masks
    generate_sam_masks(PSEUDO_LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, CHECKPOINT_PATH, MODEL_TYPE)
    
    print("\n=== Next Steps ===")
    print("Now you can run the merge command:")
    print(f'python main.py --pseudo_path "{PSEUDO_LABEL_DIR}" --sam_path "{OUTPUT_DIR}" --mode merge')