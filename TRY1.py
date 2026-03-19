# ============================================================
#   Phishing Website Detection System
#   Dataset: PhiUSIIL Phishing URL Dataset
#   Author: You
# ============================================================

# ── 0. INSTALL (run once in terminal) ───────────────────────
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna joblib

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FAST_MODE = os.getenv("FAST_MODE", "0").strip().lower() in {"1", "true", "yes", "y"}
N_TRIALS = 8 if FAST_MODE else 40
RF_ESTIMATORS = 60 if FAST_MODE else 100
BOOST_ESTIMATORS = 80 if FAST_MODE else 200
CV_SPLITS = 2 if FAST_MODE else 3

if FAST_MODE:
    print("FAST_MODE is ON: skipping plots and using lighter tuning settings.")

# ── 2. LOAD DATA ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# Replace path if needed
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

print(f"Shape         : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── 3. EDA ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: EDA")
print("=" * 60)

# 3a. Class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df, palette="Set2")
plt.title("Class Distribution (0=Legit, 1=Phishing)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
if FAST_MODE:
    plt.close()
    print("Skipped plot: class_distribution.png (FAST_MODE)")
else:
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("Saved: class_distribution.png")

# 3b. Correlation heatmap (top 20 numeric features by variance for readability)
numeric_df = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
top_features = numeric_df.var().nlargest(20).index.tolist()
plt.figure(figsize=(14, 10))
sns.heatmap(
    df[top_features + ["label"]].corr(),
    annot=False,
    cmap="coolwarm",
    linewidths=0.5,
)
plt.title("Correlation Heatmap (Top 20 Variance Features)")
plt.tight_layout()
if FAST_MODE:
    plt.close()
    print("Skipped plot: correlation_heatmap.png (FAST_MODE)")
else:
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Saved: correlation_heatmap.png")

# ── 4. PREPROCESSING ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

# Drop raw URL string column if present
drop_cols = ["URL", "url", "Domain", "domain", "Title"]
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Dropped column: {col}")

# Fill missing values with column median
df.fillna(df.median(numeric_only=True), inplace=True)

# Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# Encode any remaining categorical/text features to numeric codes
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes
    print(f"Encoded categorical columns: {cat_cols}")

print(f"Feature count : {X.shape[1]}")
print(f"Samples       : {X.shape[0]}")

# Train/test split — stratified to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size    : {X_train.shape[0]}")
print(f"Test size     : {X_test.shape[0]}")

# Scale features (needed for LR, SVM, MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 5. BASELINE MODELS ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training Baseline Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=BOOST_ESTIMATORS,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=BOOST_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # LR uses scaled data, tree models use raw
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "f1": report["weighted avg"]["f1-score"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "accuracy": report["accuracy"],
    }

    print(f"  Accuracy  : {report['accuracy']:.4f}")
    print(f"  F1 Score  : {report['weighted avg']['f1-score']:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

# ── 6. COMPARISON TABLE ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Model Comparison")
print("=" * 60)

comparison_df = pd.DataFrame(
    {
        name: {
            "Accuracy": f"{v['accuracy']:.4f}",
            "F1 Score": f"{v['f1']:.4f}",
            "Precision": f"{v['precision']:.4f}",
            "Recall": f"{v['recall']:.4f}",
            "ROC-AUC": f"{v['auc']:.4f}",
        }
        for name, v in results.items()
    }
).T

print(comparison_df.to_string())

# ── 7. CONFUSION MATRICES ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, v) in enumerate(results.items()):
    cm = confusion_matrix(y_test, v["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phishing"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(name)

plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
if FAST_MODE:
    plt.close()
    print("Skipped plot: confusion_matrices.png (FAST_MODE)")
else:
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrices.png")

# ── 8. ROC CURVES ────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for name, v in results.items():
    fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={v['auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend(loc="lower right")
plt.tight_layout()
if FAST_MODE:
    plt.close()
    print("Skipped plot: roc_curves.png (FAST_MODE)")
else:
    plt.savefig("roc_curves.png", dpi=150)
    plt.show()
    print("Saved: roc_curves.png")

# ── 9. HYPERPARAMETER TUNING (XGBoost with Optuna) ───────────
print("\n" + "=" * 60)
print("STEP 6: Hyperparameter Tuning (XGBoost via Optuna)")
print("=" * 60)

def objective(trial):
    n_estimators_low, n_estimators_high = (50, 250) if FAST_MODE else (100, 600)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", n_estimators_low, n_estimators_high),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }
    model = XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    # Use fewer CV splits in FAST_MODE for quicker iteration
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=not FAST_MODE)

print(f"\nBest ROC-AUC (CV): {study.best_value:.4f}")
print(f"Best params      : {study.best_params}")

# ── 10. FINAL MODEL ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Final Model Training with Best Params")
print("=" * 60)

best_xgb = XGBClassifier(
    **study.best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
best_xgb.fit(X_train, y_train)

y_pred_final = best_xgb.predict(X_test)
y_prob_final = best_xgb.predict_proba(X_test)[:, 1]

print("\nFinal Model Results:")
print(classification_report(y_test, y_pred_final, target_names=["Legit", "Phishing"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_final):.4f}")

# ── 11. FEATURE IMPORTANCE ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature Importance")
print("=" * 60)

feat_imp = pd.Series(best_xgb.feature_importances_, index=X.columns)
top15 = feat_imp.nlargest(15).sort_values()

plt.figure(figsize=(8, 6))
top15.plot(kind="barh", color="steelblue")
plt.title("Top 15 Feature Importances (Tuned XGBoost)")
plt.xlabel("Importance Score")
plt.tight_layout()
if FAST_MODE:
    plt.close()
    print("Skipped plot: feature_importance.png (FAST_MODE)")
else:
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Saved: feature_importance.png")
print("\nTop 15 Features:")
print(top15[::-1].to_string())

# ── 12. SAVE MODEL ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Saving Model")
print("=" * 60)

joblib.dump(best_xgb, "phishing_detector.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("Saved: phishing_detector.pkl")
print("Saved: scaler.pkl")
print("Saved: feature_columns.pkl")

# ── 13. INFERENCE EXAMPLE ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Inference Example")
print("=" * 60)

# Load and predict on a single sample
loaded_model = joblib.load("phishing_detector.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# Take one random test sample
sample = X_test.iloc[[0]][feature_cols]
prediction = loaded_model.predict(sample)[0]
confidence = loaded_model.predict_proba(sample)[0][prediction]

print(f"Prediction  : {'PHISHING' if prediction == 1 else 'LEGITIMATE'}")
print(f"Confidence  : {confidence * 100:.2f}%")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
