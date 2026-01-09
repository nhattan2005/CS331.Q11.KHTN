import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import csv
from tqdm import tqdm

def evaluate_enhanced_masks(
    enhanced_masks_dir,
    pseudo_labels_dir, 
    gt_dir,
    output_csv,
    num_cls=21
):
    """
    Đánh giá enhanced masks trực tiếp
    
    Args:
        enhanced_masks_dir: Đường dẫn đến folder enhanced masks (từ Kaggle)
        pseudo_labels_dir: Đường dẫn đến pseudo labels gốc
        gt_dir: Đường dẫn đến ground truth
        output_csv: Đường dẫn file CSV output
        num_cls: Số lượng classes (21 cho VOC)
    """
    
    # Tạo output directory
    Path(os.path.dirname(output_csv)).mkdir(parents=True, exist_ok=True)
    
    # Xóa file cũ nếu có
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    # Lấy tất cả enhanced masks
    enhanced_files = [f for f in os.listdir(enhanced_masks_dir) if f.endswith('.png')]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {len(enhanced_files)} enhanced masks...")
    print(f"{'='*60}\n")
    
    image_index = 0
    
    for enhanced_file in tqdm(enhanced_files, desc="Evaluating"):
        name = os.path.splitext(enhanced_file)[0]
        
        # Đường dẫn các files
        enhanced_path = os.path.join(enhanced_masks_dir, enhanced_file)
        gt_path = os.path.join(gt_dir, f'{name}.png')
        pseudo_path = os.path.join(pseudo_labels_dir, f'{name}.png')
        
        # Kiểm tra files tồn tại
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {name}, skipping...")
            continue
        if not os.path.exists(pseudo_path):
            print(f"Warning: Pseudo label not found for {name}, skipping...")
            continue
        
        # Load masks
        try:
            enhanced = np.array(Image.open(enhanced_path))
            gt = np.array(Image.open(gt_path))
            pseudo = np.array(Image.open(pseudo_path))
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
        
        # Calculate metrics
        cal = gt < 255
        mask_enhanced = (enhanced == gt) * cal
        mask_pseudo = (pseudo == gt) * cal
        
        P_enhanced, TP_enhanced = 0, 0
        P_pseudo, TP_pseudo = 0, 0
        T = 0
        
        IoU_enhanced, precision_enhanced, recall_enhanced = [], [], []
        IoU_pseudo, precision_pseudo, recall_pseudo = [], [], []
        
        for i in range(1, num_cls):
            true = np.sum((gt == i) * cal)
            if true == 0:
                continue
            
            # Shared
            T += true
            
            # Enhanced masks
            P_enhanced += np.sum((enhanced == i) * cal)
            TP_enhanced += np.sum((gt == i) * mask_enhanced)
            
            # Pseudo labels
            P_pseudo += np.sum((pseudo == i) * cal)
            TP_pseudo += np.sum((gt == i) * mask_pseudo)
            
            # Enhanced metrics
            IoU_enhanced.append(TP_enhanced / (T + P_enhanced - TP_enhanced + 1e-10))
            precision_enhanced.append(TP_enhanced / (P_enhanced + 1e-10))
            recall_enhanced.append(TP_enhanced / (T + 1e-10))
            
            # Pseudo metrics
            IoU_pseudo.append(TP_pseudo / (T + P_pseudo - TP_pseudo + 1e-10))
            precision_pseudo.append(TP_pseudo / (P_pseudo + 1e-10))
            recall_pseudo.append(TP_pseudo / (T + 1e-10))
        
        # Tính mean metrics
        pseudo_metrics = OrderedDict([
            ('mIoU', round(np.mean(np.array(IoU_pseudo)), 3)),
            ('mprecision', round(np.mean(np.array(precision_pseudo)), 3)),
            ('mrecall', round(np.mean(np.array(recall_pseudo)), 3))
        ])
        
        enhanced_metrics = OrderedDict([
            ('mIoU', round(np.mean(np.array(IoU_enhanced)), 3)),
            ('mprecision', round(np.mean(np.array(precision_enhanced)), 3)),
            ('mrecall', round(np.mean(np.array(recall_enhanced)), 3))
        ])
        
        # Lưu vào CSV
        update_summary(name, pseudo_metrics, enhanced_metrics, output_csv, image_index == 0)
        image_index += 1
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Processed: {image_index} images")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}\n")
    
    # In summary statistics
    print_summary_statistics(output_csv)


def update_summary(name, pseudo_metrics, enhanced_metrics, filename, write_header=False):
    """Ghi kết quả vào CSV"""
    rowd = OrderedDict(name=name)
    rowd.update([('pseudo_' + k, v) for k, v in pseudo_metrics.items()])
    rowd.update([('enhanced_' + k, v) for k, v in enhanced_metrics.items()])
    rowd.update([('mIoU_delta', enhanced_metrics['mIoU'] - pseudo_metrics['mIoU'])])
    rowd.update([('mprecision_delta', enhanced_metrics['mprecision'] - pseudo_metrics['mprecision'])])
    rowd.update([('mrecall_delta', enhanced_metrics['mrecall'] - pseudo_metrics['mrecall'])])
    
    with open(filename, mode='a', newline='') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:
            dw.writeheader()
        dw.writerow(rowd)


def print_summary_statistics(csv_path):
    """In thống kê tổng hợp"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal images: {len(df)}")
    
    print("\n--- Pseudo Labels (Original) ---")
    print(f"  Mean mIoU:       {df['pseudo_mIoU'].mean():.4f}")
    print(f"  Mean Precision:  {df['pseudo_mprecision'].mean():.4f}")
    print(f"  Mean Recall:     {df['pseudo_mrecall'].mean():.4f}")
    
    print("\n--- Enhanced Masks (After SAM) ---")
    print(f"  Mean mIoU:       {df['enhanced_mIoU'].mean():.4f}")
    print(f"  Mean Precision:  {df['enhanced_mprecision'].mean():.4f}")
    print(f"  Mean Recall:     {df['enhanced_mrecall'].mean():.4f}")
    
    print("\n--- Improvement ---")
    print(f"  Delta mIoU:      {df['mIoU_delta'].mean():.4f}")
    print(f"  Delta Precision: {df['mprecision_delta'].mean():.4f}")
    print(f"  Delta Recall:    {df['mrecall_delta'].mean():.4f}")
    
    improved = df[df['mIoU_delta'] > 0]
    worse = df[df['mIoU_delta'] < 0]
    same = df[df['mIoU_delta'] == 0]
    
    print(f"\n--- Distribution ---")
    print(f"  Improved:  {len(improved):4d} ({len(improved)/len(df)*100:5.1f}%)")
    print(f"  Worse:     {len(worse):4d} ({len(worse)/len(df)*100:5.1f}%)")
    print(f"  Same:      {len(same):4d} ({len(same)/len(df)*100:5.1f}%)")
    
    if len(improved) > 0:
        print(f"\n  Avg improvement (improved cases): {improved['mIoU_delta'].mean():.4f}")
    if len(worse) > 0:
        print(f"  Avg decline (worse cases):        {worse['mIoU_delta'].mean():.4f}")
    
    print("\n--- Top 5 Best Improvements ---")
    top5 = df.nlargest(5, 'mIoU_delta')[['name', 'pseudo_mIoU', 'enhanced_mIoU', 'mIoU_delta']]
    for idx, row in top5.iterrows():
        print(f"  {row['name']:15s}: {row['pseudo_mIoU']:.3f} → {row['enhanced_mIoU']:.3f} (Δ {row['mIoU_delta']:+.3f})")
    
    print("\n--- Top 5 Worst Cases ---")
    worst5 = df.nsmallest(5, 'mIoU_delta')[['name', 'pseudo_mIoU', 'enhanced_mIoU', 'mIoU_delta']]
    for idx, row in worst5.iterrows():
        print(f"  {row['name']:15s}: {row['pseudo_mIoU']:.3f} → {row['enhanced_mIoU']:.3f} (Δ {row['mIoU_delta']:+.3f})")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Cấu hình
    ENHANCED_MASKS_DIR = "../enhanced_masks"  # Folder đã download từ Kaggle
    PSEUDO_LABELS_DIR = "pseudo_labels/transcam"
    GT_DIR = "../archive/VOC2012_train_val/VOC2012_train_val/SegmentationClass"
    OUTPUT_CSV = "evals/enhanced_masks_evaluation.csv"
    
    # Kiểm tra đường dẫn
    if not os.path.exists(ENHANCED_MASKS_DIR):
        print(f"ERROR: Enhanced masks not found at: {ENHANCED_MASKS_DIR}")
        print("Please specify the correct path to your downloaded enhanced_masks folder")
        sys.exit(1)
    
    if not os.path.exists(PSEUDO_LABELS_DIR):
        print(f"ERROR: Pseudo labels not found at: {PSEUDO_LABELS_DIR}")
        sys.exit(1)
    
    if not os.path.exists(GT_DIR):
        print(f"ERROR: Ground truth not found at: {GT_DIR}")
        sys.exit(1)
    
    # Chạy evaluation
    evaluate_enhanced_masks(
        enhanced_masks_dir=ENHANCED_MASKS_DIR,
        pseudo_labels_dir=PSEUDO_LABELS_DIR,
        gt_dir=GT_DIR,
        output_csv=OUTPUT_CSV,
        num_cls=21
    )