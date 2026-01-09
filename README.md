# Weakly Supervised Semantic Segmentation (WSSS) with SAM + CAM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)](https://streamlit.io/)

> **Final Project – Advanced Computer Vision (CS331.Q11.KHTN)**  
> **University of Information Technology – VNU-HCM**

---

## 📖 Introduction

This project focuses on **Weakly Supervised Semantic Segmentation (WSSS)**, a task that aims to reduce the dependency on expensive pixel-level annotations by using only **image-level labels**.

We propose a framework that combines:

- **TransCAM** (Transformer-based Class Activation Mapping)
- **Segment Anything Model (SAM)**

to address two major limitations of traditional CAM-based WSSS methods:

1. **Partial Activation:** Only the most discriminative regions of an object are activated.
2. **False Activation:** CAM responses often leak into background regions.

By integrating SAM masks with CAM-based pseudo labels, the proposed method generates more complete and accurate supervision for semantic segmentation.

---

## 🚀 Features

- **SAM + CAM Integration**  
  Combines SAM-generated masks with CAM-based pseudo labels to improve object coverage and boundary quality.

- **Enhanced Pseudo Labels**  
  Multiple mask merging strategies based on IoU and confidence refinement.

- **DeepLabV3+ Training**  
  Trains a DeepLabV3+ segmentation model using enhanced pseudo labels.

- **Comprehensive Evaluation**  
  Supports mIoU, precision, and recall evaluation for both pseudo labels and final segmentation results.

- **Web Demo**  
  Interactive Streamlit-based application for inference and visualization.

---

## 📂 Project Structure

CS331.Q11.KHTN
├── check_pseudo_label.py # Analyze pseudo label quality
├── evaluate_enhanced.py # Evaluate enhanced masks
├── generate_sam_masks.py # Generate masks using SAM
├── main.py # Main pipeline (merge, evaluate, visualize)
├── processor.py # Core processing logic
├── train_deeplabv3.py # Train DeepLabV3+ model
├── merge/ # Mask merging strategies
│ ├── max_iou.py
│ ├── max_iou_imp.py
│ ├── max_iou_imp2.py
│ ├── merge_base.py
│ └── merge_customize.py
├── util/
│ └── vis.py # Visualization utilities
├── Web_demo/ # Streamlit web demo
│ ├── src/
│ │ ├── app.py # Main Streamlit app
│ │ ├── model/ # Model loading and inference
│ │ ├── preprocessing/ # Preprocessing utilities
│ │ └── utils/ # Helper functions
│ └── requirements.txt # Web demo dependencies
└── README.md

---

## 📊 Experimental Results

### 🔹 Pseudo Label Quality

| Method                        | mIoU   |
|------------------------------|--------|
| TransCAM (Original)          | 63.16% |
| **TransCAM + SAM (Proposed)**| **65.85%** |

### 🔹 DeepLabV3+ Segmentation Performance

| Configuration                         | Accuracy | mIoU   |
|--------------------------------------|----------|--------|
| DeepLabV3+ + Original Pseudo Mask    | 89.27%   | 51.21% |
| **DeepLabV3+ + Enhanced Mask**       | **90.17%** | **52.29%** |

---

## 💻 Web Demo

The project includes a **Streamlit-based web demo** that performs end-to-end semantic segmentation without requiring any user prompts.

### Demo Features

- Upload input images (JPG, PNG)
- Automatic class detection and segmentation
- Visualization of:
  - Original Image
  - Segmentation Mask
  - Overlayed Result (Mask + Image)
