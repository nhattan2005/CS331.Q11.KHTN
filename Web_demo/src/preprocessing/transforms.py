from PIL import Image
import numpy as np
import torch

def preprocess_image(image, target_size=(384, 384)):
    """
    Preprocess image for WSSS model inference.
    
    Steps:
    1. Resize to 384x384
    2. Convert to float32, divide by 255.0
    3. Subtract mean: [0.485, 0.456, 0.406] (NO std division)
    4. Permute to (C, H, W)
    
    Args:
        image: PIL Image
        target_size: Tuple of (height, width)
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Subtract mean (NO std division as per requirements)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_array = img_array - mean
    
    # Permute to (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_array)
    
    return img_tensor


def get_pascal_voc_colormap():
    """Generate Pascal VOC colormap for 21 classes."""
    colormap = np.zeros((21, 3), dtype=np.uint8)
    
    colormap[0] = [0, 0, 0]          # background
    colormap[1] = [128, 0, 0]        # aeroplane
    colormap[2] = [0, 128, 0]        # bicycle
    colormap[3] = [128, 128, 0]      # bird
    colormap[4] = [0, 0, 128]        # boat
    colormap[5] = [128, 0, 128]      # bottle
    colormap[6] = [0, 128, 128]      # bus
    colormap[7] = [128, 128, 128]    # car
    colormap[8] = [64, 0, 0]         # cat
    colormap[9] = [192, 0, 0]        # chair
    colormap[10] = [64, 128, 0]      # cow
    colormap[11] = [192, 128, 0]     # dining table
    colormap[12] = [64, 0, 128]      # dog
    colormap[13] = [192, 0, 128]     # horse
    colormap[14] = [64, 128, 128]    # motorbike
    colormap[15] = [192, 128, 128]   # person
    colormap[16] = [0, 64, 0]        # potted plant
    colormap[17] = [128, 64, 0]      # sheep
    colormap[18] = [0, 192, 0]       # sofa
    colormap[19] = [128, 192, 0]     # train
    colormap[20] = [0, 64, 128]      # tv/monitor
    
    return colormap


def get_class_names():
    """Get Pascal VOC class names mapping."""
    class_names = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "dining table",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "potted plant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tv/monitor"
    }
    return class_names


def colorize_mask(mask):
    """
    Convert segmentation mask to colored image using Pascal VOC colormap.
    
    Args:
        mask: numpy array of shape (H, W) with class indices
    
    Returns:
        numpy array of shape (H, W, 3) with RGB colors
    """
    colormap = get_pascal_voc_colormap()
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for label in range(21):
        color_mask[mask == label] = colormap[label]
    
    return color_mask


def overlay_mask_on_image(original_image, colored_mask, alpha=0.5):
    """
    Overlay colored segmentation mask on original image.
    
    Args:
        original_image: PIL Image (original image)
        colored_mask: numpy array (H, W, 3) with RGB colors
        alpha: float, transparency of the overlay (0.0 to 1.0)
    
    Returns:
        numpy array of blended image
    """
    # Convert PIL image to numpy array
    img_array = np.array(original_image.resize((colored_mask.shape[1], colored_mask.shape[0])))
    
    # Ensure both arrays have the same shape
    if img_array.shape[:2] != colored_mask.shape[:2]:
        # Resize colored_mask to match image size
        colored_mask_pil = Image.fromarray(colored_mask)
        colored_mask_pil = colored_mask_pil.resize((img_array.shape[1], img_array.shape[0]))
        colored_mask = np.array(colored_mask_pil)
    
    # Blend images
    blended = (alpha * colored_mask + (1 - alpha) * img_array).astype(np.uint8)
    
    return blended