# 🚗🚋 Audio Classification: Car vs Tram using MFCC and SVM

This project implements a **binary audio classification system** to distinguish between **car** and **tram** sounds using **MFCC features** and a **Support Vector Machine (SVM)** classifier.
The pipeline follows standard machine learning practice with **user-disjoint train/validation/test splits** and includes experiments on **training set size vs accuracy**.

---

## 📌 Project Overview

The system performs the following steps:

1. **Audio preprocessing**

   * Convert audio to `.wav`
   * Trim or pad to fixed length
   * Normalize amplitude

2. **Feature extraction**

   * Extract MFCC features
   * Compute mean and standard deviation per audio file

3. **Dataset construction**

   * Create feature matrices (`X`) and labels (`y`)
   * Split data into **train**, **validation**, and **test** sets using **disjoint users**

4. **Model training**

   * Train SVM classifiers using a scikit-learn pipeline
   * Train multiple models with increasing training set sizes (100, 200, 300, …)

5. **Evaluation**

   * Evaluate using accuracy, precision, recall
   * Visualize confusion matrices
   * Plot **accuracy vs number of training samples**

---

## 🧠 Features Used

* **MFCC (Mel-Frequency Cepstral Coefficients)**
* Mean and standard deviation of MFCCs per audio file

These features capture perceptually meaningful spectral characteristics of audio signals.

---

## 📁 Project Structure

```
project_root/
│
├── data/
│   ├── raw/                  # Original audio files (per user)
│   ├── processed/            # Trimmed & normalized audio
│   ├── train_dataset/
│   │   ├── x_dataset.npy
│   │   └── y_dataset.npy
│   ├── val_dataset/
│   │   ├── x_dataset.npy
│   │   └── y_dataset.npy
│   └── test_dataset/
│       ├── x_dataset.npy
│       └── y_dataset.npy
│
├── models/
│   ├── svm_100.pkl
│   ├── svm_200.pkl
│   ├── svm_300.pkl
│   └── ...
│
├── scripts/
│   ├── prepare_data.py       # Audio trimming & normalization
│   ├── extract_features.py   # MFCC feature extraction
│   ├── train.py              # Model training
│   ├── val.py                # Validation evaluation
│   ├── test.py               # Final test evaluation
│   └── accuracy_plot.py      # Accuracy vs training size
│
├── results/
│   ├── confusion_matrix_val.png
│   ├── confusion_matrix_test.png
│   └── accuracy_vs_training_samples.png
│
└── README.md
```

---

## ⚙️ Requirements

* Python 3.8+
* NumPy
* Librosa
* SoundFile
* scikit-learn
* Matplotlib
* Seaborn
* Joblib

Install dependencies with:

```bash
pip install numpy librosa soundfile scikit-learn matplotlib seaborn joblib
```

---

## 🚀 How to Run

### 1️⃣ Preprocess audio

```bash
python scripts/prepare_data.py
```

### 2️⃣ Extract features and create datasets

```bash
python scripts/extract_features.py
```

### 3️⃣ Train models

```bash
python scripts/train.py
```

This will generate multiple models:

```
models/svm_100.pkl
models/svm_200.pkl
...
```

### 4️⃣ Validate model

```bash
python scripts/val.py
```

### 5️⃣ Test model

```bash
python scripts/test.py
```

### 6️⃣ Plot accuracy vs training size

```bash
python scripts/accuracy_plot.py
```

---

## 📊 Evaluation Metrics

The following metrics are used:

* **Accuracy**
* **Precision**
* **Recall**
* **Confusion Matrix** (visualized as heatmaps)

---

## 📈 Experimental Analysis

Multiple SVM models are trained using progressively larger **balanced subsets** of the training data.
This allows analysis of how **training data size impacts model performance**.

**Accuracy vs Training Samples** is plotted to study:

* Data efficiency
* Performance saturation
* Generalization behavior

---

## 🧪 Dataset Split Strategy

* **Training set**: Used to train models
* **Validation set**: Used for model comparison and development
* **Test set**: Used only once for final performance reporting

All splits are **user-disjoint** to prevent data leakage.

---

## 📝 Notes

* Class labels:

  * `0` → Car
  * `1` → Tram
* Feature scaling is handled inside the model pipeline
* Random seeds are fixed for reproducibility

---

## 📚 License / Academic Use

This project is intended for **educational and academic purposes**.

---
