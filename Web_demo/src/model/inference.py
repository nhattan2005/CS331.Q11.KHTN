import torch
import numpy as np
from preprocessing.transforms import preprocess_image, colorize_mask

def run_inference(model, image, device):
    """
    Run inference on a single image.
    
    Args:
        model: Loaded PyTorch model
        image: PIL Image
        device: Device to run inference on (can be string or torch.device)
    
    Returns:
        Tuple of (colored_mask, segmentation_map)
        - colored_mask: numpy array (H, W, 3) with RGB colors
        - segmentation_map: numpy array (H, W) with class indices
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Add batch dimension and move to device
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Get segmentation map (argmax over classes)
    segmentation_map = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Colorize the mask
    colored_mask = colorize_mask(segmentation_map)
    
    return colored_mask, segmentation_map