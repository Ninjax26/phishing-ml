# Phishing Website Detection System

Machine learning pipeline for phishing URL detection using the PhiUSIIL dataset.

## Project File

- `TRY1.py`: End-to-end workflow (EDA, preprocessing, baseline models, Optuna tuning, final model save, inference example).

## Requirements

Use a Python virtual environment, then install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna joblib
```

## Dataset

Expected file in project root:

- `PhiUSIIL_Phishing_URL_Dataset.csv`

If you have `PhiUSIIL_Phishing_URL_Dataset.csv.zip`, extract it first:

```bash
unzip -o "PhiUSIIL_Phishing_URL_Dataset.csv.zip"
```

## Run

Full pipeline:

```bash
python TRY1.py
```

Fast iteration mode (fewer Optuna trials, lighter estimators, skips plots):

```bash
FAST_MODE=1 python TRY1.py
```

## Outputs

The script saves:

- `phishing_detector.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

In non-fast mode, it also saves plots:

- `class_distribution.png`
- `correlation_heatmap.png`
- `confusion_matrices.png`
- `roc_curves.png`
- `feature_importance.png`
