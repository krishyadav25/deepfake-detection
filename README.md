# 🔍 Multimodal Deepfake Detection System

> Deep Learning model for detecting AI-generated fake images using 
> a 4-branch parallel CNN architecture with Attention Gate.

---

## 📋 Project Overview

This project implements a **multimodal deepfake detection system** 
that detects fake images using generation-level forgery traces 
including FFT artifacts, edge inconsistencies, semantic features, 
and acquisition traces.

**Developed by:**

1. Krish Yadav
2. Pranjal Seluriyal
3. Isha

**Institution:** School of Computer Science, UPES Dehradun

---

## 🏗️ Model Architecture

<img width="896" height="1177" alt="image" src="https://github.com/user-attachments/assets/740dbd27-ffec-42a3-be1f-aa1e45d9e3e1" />

---

## 📊 Results

| Metric | REAL | FAKE | Macro Avg |
|---|---|---|---|
| Precision | 0.9837 | 0.9188 | 0.9513 |
| Recall | 0.8522 | 0.9916 | 0.9219 |
| F1-Score | 0.9132 | 0.9538 | 0.9335 |
| Accuracy | — | — | 0.9397 |
| ROC-AUC | — | — | 0.9645 |

**Best Threshold: 0.45**

---

## 📁 Datasets Used

| Dataset | Modality | Purpose |
|---|---|---|
| 140k Real and Fake Faces | Image | StyleGAN fake faces |
| FaceForensics++ | Image/Video | 6 manipulation methods |
| CelebA | Image | Diverse real faces |
| ciplab Real and Fake | Image | Additional diversity |

---

## 🔧 Tech Stack

- **Framework:** PyTorch
- **Language:** Python 3.8+
- **Platform:** Kaggle (GPU T4 x2)
- **Key Libraries:** OpenCV, librosa, scikit-learn

---

## 🚀 How to Run

### On Kaggle
1. Open the notebook on Kaggle
2. Add datasets:
   - 140k Real and Fake Faces
   - FaceForensics++ (ff-c23)
   - CelebA Dataset
   - Real and Fake Face Detection (ciplab)
3. Enable GPU (T4 x2)
4. Run all cells

### Inference on Single Image
```python
predict_my_image("path/to/image.jpg", threshold=0.45)
```

---

## 📈 Training Curves

![Training Curves](assets/training_curves.png)

---

## 📊 Evaluation Plots

![Evaluation](assets/evaluation_plots.png)

---

## 📋 Evaluation Tables

![Metrics Table](assets/table_II_metrics.png)
![Class Report](assets/table_III_class_report.png)
![Score Distribution](assets/table_IV_score_dist.png)
![Threshold Analysis](assets/table_V_threshold.png)
![Confusion Matrix Breakdown Table](assets/table_VI_confusion.png)

---

## 🏛️ Project Structure
deepfake-detection/

├── deepfake-detection.ipynb   # Main notebook

├── requirements.txt           # Dependencies

├── README.md                  # This file

└── assets/                    # Evaluation plots & tables

---

## 📚 References

1. Y. Ma et al., "Deep learning technology for face forgery 
   detection: A survey," Neurocomputing, 2024.
2. A. Rossler et al., "FaceForensics++: Learning to Detect 
   Manipulated Facial Images," ICCV, 2019.
3. B. Dolhansky et al., "The DeepFake Detection Challenge 
   Dataset," arXiv:2006.07397, 2020.
