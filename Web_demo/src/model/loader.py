import torch
import segmentation_models_pytorch as smp
import os

def load_wsss_model(model_path, device=None):
    """
    Load DeepLabV3Plus model for WSSS task.
    
    Args:
        model_path: Path to the trained model.pth file
        device: Device to load the model on (cuda/cpu). If None, auto-detect.
    
    Returns:
        Tuple of (model, device)
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Create DeepLabV3Plus model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,  # No pretrained weights
        classes=21,  # Pascal VOC classes
        activation=None
    )
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different saving formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            elif 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                # Assume the dict itself is the state dict
                model.load_state_dict(state_dict)
        else:
            # If it's not a dict, it might be the model itself
            model = state_dict
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {str(e)}")
    
    model = model.to(device)
    model.eval()
    
    return model, device