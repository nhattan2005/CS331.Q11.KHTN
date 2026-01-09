# Weakly Supervised Semantic Segmentation (WSSS) with SAM + CAM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)](https://streamlit.io/)

> **Final Project for Advanced Computer Vision (CS331.Q11.KHTN)**  
> **University of Information Technology - VNU-HCM**

---

## 📖 Introduction

This project addresses the problem of **Weakly Supervised Semantic Segmentation (WSSS)**, which aims to reduce the cost of pixel-level annotations by leveraging image-level labels.

The system combines **TransCAM** (based on Transformer Attention) and **Segment Anything Model (SAM)** to overcome the limitations of traditional CAM-based methods:

1. **Partial Activation:** Only the most distinctive parts of objects are detected.
2. **False Activation:** Activation spreads to background regions.

---

## 🚀 Features

- **SAM + CAM Integration:**  
  Combines SAM-generated masks with CAM-based pseudo labels to enhance segmentation quality.

- **DeepLabV3+ Training:**  
  Trains a DeepLabV3+ model on enhanced masks for improved segmentation performance.

- **Evaluation Metrics:**  
  Calculates mIoU, precision, and recall for both pseudo labels and enhanced masks.

- **Web Demo:**  
  Provides an interactive Streamlit-based web application for inference and visualization.

---

## 📂 Project Structure

```
CS331.Q11.KHTN
├── check_pseudo_label.py           # Analyze pseudo labels
├── evaluate_enhanced.py            # Evaluate enhanced masks
├── generate_sam_masks.py           # Generate SAM masks
├── main.py                         # Main entry point for merging, evaluation, and visualization
├── processor.py                    # Core processing logic
├── train_deeplabv3.py              # Train DeepLabV3+ model
├── merge/                          # Mask merging strategies
│   ├── max_iou.py
│   ├── max_iou_imp.py
│   ├── max_iou_imp2.py
│   ├── merge_base.py
│   └── merge_customize.py
├── util/                           # Utility functions
│   └── vis.py
├── Web_demo/                       # Streamlit-based web demo
│   ├── src/
│   │   ├── app.py                  # Main Streamlit app
│   │   ├── model/                  # Model loading and inference
│   │   ├── preprocessing/          # Preprocessing utilities
│   │   └── utils/                  # Helper functions
│   └── requirements.txt            # Dependencies for the web demo
└── README.md                       # Project documentation
```

---

## 📊 Experimental Results

### Pseudo Label Quality

| Method                      | mIoU     |
|-----------------------------|----------|
| TransCAM (Original)         | 63.16%   |
| **TransCAM + SAM (Proposed)** | **65.85%** |

### DeepLabV3+ Segmentation Performance

| Configuration                   | Accuracy    | mIoU     |
|----------------------------------|------------|----------|
| DeepLabV3+ + Original Pseudo Mask| 89.27%     | 51.21%   |
| **DeepLabV3+ + Enhanced Mask**   | **90.17%** | **52.29%** |

---

## 💻 Web Demo

The project includes a **Streamlit-based web demo** for end-to-end segmentation from input images without requiring any prompts.

### Features:

- Upload images (JPG, PNG).
- Automatic segmentation and class detection.
- Visualization of:
  - Original Image
  - Segmentation Mask
  - Overlayed Image (Mask + Original)

### How to Run the Demo:

1. Navigate to the Web_demo directory:
   ```bash
   cd Web_demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

---

## 🛠️ How to Use

**1. Generate SAM Masks**  
Run the `generate_sam_masks.py` script to generate SAM masks:
```bash
python generate_sam_masks.py
```

**2. Merge Masks**  
Use the `main.py` script to merge pseudo labels and SAM masks:
```bash
python main.py --mode merge
```

**3. Evaluate Enhanced Masks**  
Evaluate the enhanced masks using:
```bash
python main.py --mode eval
```

**4. Train DeepLabV3+**  
Train the DeepLabV3+ model with enhanced masks:
```bash
python train_deeplabv3.py
```
