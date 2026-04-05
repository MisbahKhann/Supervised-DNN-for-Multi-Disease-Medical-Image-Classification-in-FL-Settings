# Federated Learning for Medical Image Classification
### Reproduction of Gamal et al. (2025) — ANN Assignment 2
**Authors:** Misbah Khan (23i-0101) | Wajeeha Mahmood (23i-0105)  
**Course:** Artificial Neural Networks — FAST-NUCES, AI-A

---

## Overview
This repository reproduces the Privacy-Enhanced Federated Learning framework with Adaptive Aggregation proposed by Gamal et al. (2025), published in the Journal of Big Data. The framework trains a ResNet50 model across multiple simulated hospital clients without sharing raw patient data, using an adaptive aggregation strategy that dynamically switches between FedAvg and FedSGD based on client divergence.

---

## Datasets
Download all three datasets from Kaggle and place them in a `data/` folder:

| Dataset | Kaggle Link | Classes |
|---|---|---|
| TB Chest X-ray | tawsifurrahman/tuberculosis-tb-chest-xray-dataset | 2 |
| Brain Tumor MRI | sartajbhuvaji/brain-tumor-classification-mri | 4 |
| Diabetic Retinopathy | sovitrath/diabetic-retinopathy-224x224-2019-data | 5 |

Expected folder structure after extraction:
```
data/
  TB_Chest_Radiography_Database/
      Normal/
      Tuberculosis/
  Training/
      glioma_tumor/
      meningioma_tumor/
      no_tumor/
      pituitary_tumor/
  Testing/
      (same 4 classes)
  colored_images/
      Mild/
      Moderate/
      No_DR/
      Proliferate_DR/
      Severe/
```

---

## Requirements
```
torch
torchvision
timm
scikit-learn
tqdm
numpy
```
Install with:
```bash
pip install torch torchvision timm scikit-learn tqdm numpy
```

---

## Usage

**On Google Colab (recommended):**
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Run
exec(open("fed_adaptive_medical.py").read())
```

**Change dataset in Config at bottom of file:**
```python
cfg = Config(
    dataset_name = "brain_tumor",        # or "tb_xray" / "diabetic_retinopathy"
    data_root    = "/content/drive/MyDrive/fl_data",
    save_dir     = "/content/drive/MyDrive/fl_checkpoints",
    backbone     = "resnet50",
    num_rounds   = 40,
    num_clients  = 5,
    lr           = 8e-4,
)
```

---

## Results

| Dataset | Paper Accuracy | Reproduced Accuracy |
|---|---|---|
| Brain Tumor MRI | 96.3% | 61.42% |
| TB Chest X-ray | ~95.1% | 83.35% |
| Diabetic Retinopathy | ~94.7% | 66.97% |

---

## Experiment Logs
Round-by-round logs saved as CSV after each run:
- `logs_brain_tumor.csv`
- `logs_tb.csv`
- `logs_dr.csv`

---

## Reference
Gamal, A.M., Houssein, E.H., Younis, E.M.G. & Mohamed, E. (2025). A Privacy-Enhanced Framework for Collaborative Big Data Analysis in Healthcare Using Adaptive Federated Learning Aggregation. *Journal of Big Data*, Springer. DOI: 10.1186/s40537-025-01169-8
