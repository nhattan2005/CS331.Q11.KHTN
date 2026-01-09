# Giải quyết bài toán Weakly Supervised Semantic Segmentation (WSSS) bằng phương pháp SAM + CAM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)](https://streamlit.io/)

> **Đồ án cuối kỳ môn Thị giác máy tính nâng cao (CS331.Q11.KHTN)** > **Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM**

## Giới thiệu

Project này tập trung giải quyết bài toán **Phân đoạn ngữ nghĩa giám sát yếu (WSSS)**, nhằm giảm thiểu chi phí gán nhãn dữ liệu bằng cách sử dụng nhãn cấp hình ảnh (image-level labels) thay vì nhãn cấp điểm ảnh (pixel-level labels).

Hệ thống sử dụng quy trình kết hợp giữa **TransCAM** (dựa trên Transformer Attention) và **Segment Anything Model (SAM)** để khắc phục hai hạn chế lớn của phương pháp CAM truyền thống:
1.  **Partial Activation:** Kích hoạt cục bộ (chỉ nhận diện phần đặc trưng nhất của đối tượng).
2.  **False Activation:** Kích hoạt sai (lan ra vùng nền).

## Kết quả thực nghiệm

Thực nghiệm được tiến hành trên tập dữ liệu **PASCAL VOC 2012** sử dụng GPU P100.

### Chất lượng Nhãn giả (Pseudo Labels)
| Phương pháp | mIoU |
|:---|:---:|
| Pseudo mask từ TransCAM gốc | 63.16% |
| **Pseudo mask từ TransCAM + SAM (Đề xuất)** | **65.85%** |

### Hiệu năng mô hình phân đoạn (DeepLabV3+)
Kết quả trên tập Validation PASCAL VOC 2012:

| Cấu hình | Accuracy | mIoU |
|:---|:---:|:---:|
| DeepLabV3 + Pseudo_mask gốc | 89.27% | 51.21% |
| **DeepLabV3 + Enhanced_mask (Đề xuất)** | **90.17%** | **52.29%** |

## 💻 Demo Ứng dụng

Dự án bao gồm một Web Demo xây dựng bằng **Streamlit**, cho phép thực hiện phân đoạn end-to-end từ ảnh đầu vào mà không cần bất kỳ gợi ý (prompt) nào.

**Tính năng:**
* Upload ảnh (JPG, PNG).
* Tự động phân đoạn và nhận diện lớp.
* Hiển thị trực quan: Ảnh gốc, Mask phân đoạn, và Ảnh chồng lớp (Overlay).
