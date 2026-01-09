from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def overlay_segmentation(image_path, segmentation_map):
    image = Image.open(image_path).convert("RGB")
    segmentation_map = Image.fromarray(segmentation_map.astype(np.uint8))

    # Create a color overlay for the segmentation map
    color_overlay = np.zeros((*segmentation_map.size, 3), dtype=np.uint8)
    color_overlay[segmentation_map == 1] = [255, 0, 0]  # Red for class 1
    color_overlay[segmentation_map == 2] = [0, 255, 0]  # Green for class 2
    color_overlay[segmentation_map == 3] = [0, 0, 255]  # Blue for class 3

    # Blend the original image with the color overlay
    blended_image = Image.blend(image, Image.fromarray(color_overlay), alpha=0.5)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Segmentation Map")
    plt.imshow(segmentation_map, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(blended_image)
    plt.axis('off')

    plt.show()