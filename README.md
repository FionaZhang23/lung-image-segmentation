# lung-image-segmentation

## 📌 Overview

This project implements a deep learning-based segmentation model to classify suspicious regions in radiological images of human lungs. The model performs pixel-wise classification with 4 possible classes and is trained using historical lung scans labeled by radiologists.

This work was completed as part of **CSC373/673: Data Mining (Spring 2025)** at Wake Forest University.

> ⚠️ **Note**: Due to privacy and medical data protection concerns, the image data used in this project is not publicly available.

---

## 🧠 Objective

The primary goal is to **predict per-pixel region labels** (0, 1, 2, or 3) in lung images using supervised learning. These labels are stored in **one-hot encoded format** (e.g., `[1 0 0 0]` = class 0, `[0 0 0 1]` = class 3).

Classes:
- 0 and 1: Suspicious regions requiring radiologist attention
- 2 and 3: Normal regions

---

## 🎯 Deliverables

| File                                | Description                                     |
|-------------------------------------|-------------------------------------------------|
| `output/predictions.npy`           | Prediction masks (100 test scans)              |
| `output/modeling_pipeline.pkl`     | Saved segmentation pipeline (`joblib`)         |
| `output/report.pdf`                | Report documenting data cleaning, modeling, evaluation |
| `scripts/assignment_4.py`          | Main entry script to train and run the model   |
| *(+ additional modules)*           | Feature engineering, quality checking, etc.    |

---

## 🧠 Methodology

| Phase | Task                                     |
|-------|------------------------------------------|
| I     | Preprocess and normalize raw scan data   |
| II    | Convert one-hot encoded labels to numeric format |
| III   | Train classification model (per-pixel)   |
| IV    | Evaluate accuracy using labeled masks    |
| V     | Export pipeline and prediction masks     |

---

## 🗂️ Project Structure

```bash
scripts/                          # All Python scripts
├── __pycache__/                  # Cache files
├── assignment_4.py               # Main training & prediction script
├── check_quality.py              # Data checker (optional)
├── check.py                      # Evaluation utilities
├── custom_classifier.py          # Custom per-pixel classifier logic
├── preprocess_data.py            # One-hot decoding, reshaping, etc.
└── utils.py                      # Helper functions

output/                           # Model artifacts & results
├── image.png                     # Sample prediction visualization
├── label_distribution.png        # Label histogram or breakdown
└── report.pdf                    # Project summary and methodology
