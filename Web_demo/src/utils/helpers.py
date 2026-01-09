import os
import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def postprocess_output(output):
    output = output.squeeze(0).detach().cpu().numpy()
    output = output.argmax(axis=0)  # Get the class with the highest score
    return output

def save_output_image(output, save_path):
    output_image = Image.fromarray(output.astype('uint8'))
    output_image.save(save_path)