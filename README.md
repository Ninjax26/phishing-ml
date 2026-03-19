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

Dataset used:

- `PhiUSIIL Phishing URL Dataset`

Expected file in project root:

- `PhiUSIIL_Phishing_URL_Dataset.csv`

### Download Options

Option 1 (recommended): use the dataset zip included in this repository:

- `PhiUSIIL_Phishing_URL_Dataset.csv.zip`

Then extract it in the project root:

```bash
unzip -o "PhiUSIIL_Phishing_URL_Dataset.csv.zip"
```

Option 2: download the same dataset separately and place the CSV in the project root as:

- `PhiUSIIL_Phishing_URL_Dataset.csv`

If you have a zip file, extract it first:

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
