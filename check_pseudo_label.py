import numpy as np
from PIL import Image
import os

def check_pseudo_label(pseudo_path, image_name):
    """
    Kiểm tra pseudo label chứa những class nào
    """
    pseudo_file = os.path.join(pseudo_path, f"{image_name}.png")
    
    if not os.path.exists(pseudo_file):
        print(f"File không tồn tại: {pseudo_file}")
        return
    
    # Đọc pseudo label
    pseudo = np.array(Image.open(pseudo_file))
    
    # Lấy các class unique
    unique_classes = np.unique(pseudo)
    
    # VOC classes
    voc_classes = {
        0: 'background',
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
        9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
        13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor',
        255: 'ignore'
    }
    
    print(f"\n=== Phân tích pseudo label: {image_name} ===")
    print(f"Shape: {pseudo.shape}")
    print(f"Classes có trong pseudo label:")
    
    for cls_id in unique_classes:
        cls_name = voc_classes.get(cls_id, f'unknown_{cls_id}')
        pixel_count = np.sum(pseudo == cls_id)
        percentage = (pixel_count / pseudo.size) * 100
        print(f"  Class {cls_id} ({cls_name}): {pixel_count} pixels ({percentage:.2f}%)")
    
    # Visualize
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(pseudo, cmap='tab20')
    plt.colorbar()
    plt.title(f'Pseudo Label: {image_name}')
    plt.savefig(f'pseudo_label_check_{image_name}.png')
    print(f"\nĐã lưu visualization: pseudo_label_check_{image_name}.png")

# Kiểm tra file cụ thể
check_pseudo_label(
    "D:\\SAM2CAM\\SEPL\\SAM_WSSS\\processed_masks\\transcam\\transcam_voc12_ver2_max_iou_imp2",
    "2007_000032"
)