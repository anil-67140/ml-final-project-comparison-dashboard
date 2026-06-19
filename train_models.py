"""
train_models.py
===============
Run this FIRST before launching Flask or Streamlit.

    python train_models.py

What it does:
  1. Loads Students_Performance_Dataset.csv
  2. Cleans and preprocesses the data
  3. Trains Logistic Regression + Random Forest
  4. Evaluates both models and saves all metrics
  5. Saves models + artefacts into ./models/ folder

After running you will see:
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/scaler.pkl
    models/imputer.pkl
    models/metrics.json
    models/feature_importance.json
    models/roc_data.json
    models/dataset_summary.json
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH      = "Students_Performance_Dataset.csv"
MODELS_DIR    = "models"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CLASS_NAMES   = ["A", "B", "C", "D", "F"]   # GradeClass 0=A 1=B 2=C 3=D 4=F

# These 12 columns are the features we use for prediction.
# We DROP: StudentID (just an ID), GPA (would leak the target because
# GradeClass is literally computed from GPA thresholds).
FEATURE_COLS = [
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering",
]

FEATURE_LABELS = {
    "Age":               "Age (years)",
    "Gender":            "Gender",
    "Ethnicity":         "Ethnicity",
    "ParentalEducation": "Parental Education",
    "StudyTimeWeekly":   "Weekly Study Time (hrs)",
    "Absences":          "Absences (days/year)",
    "Tutoring":          "Tutoring",
    "ParentalSupport":   "Parental Support",
    "Extracurricular":   "Extracurricular",
    "Sports":            "Sports",
    "Music":             "Music",
    "Volunteering":      "Volunteering",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Student Performance ML — Training Script")
print("=" * 60)

print(f"\n[1] Loading dataset from '{CSV_PATH}' ...")
df = pd.read_csv(CSV_PATH)
print(f"    Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Columns  : {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Preprocessing ...")

# 2a. Missing values — only in StudyTimeWeekly and Absences
missing_before = df[["StudyTimeWeekly", "Absences"]].isnull().sum().sum()
imputer = SimpleImputer(strategy="median")
df[["StudyTimeWeekly", "Absences"]] = imputer.fit_transform(
    df[["StudyTimeWeekly", "Absences"]]
)
print(f"    Missing values imputed (median): {missing_before} values filled")

# 2b. Remove duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"    Duplicates removed: {before - len(df)}")

# 2c. Separate features and target
#     GPA is dropped to PREVENT DATA LEAKAGE.
#     GradeClass = 0 if GPA>=3.5, 1 if GPA>=3.0, etc.
#     If we include GPA the model trivially gets ~100% accuracy.
X = df[FEATURE_COLS].copy().astype(float)
y = df["GradeClass"].values.astype(int)

print(f"    Feature matrix: {X.shape}")
print(f"    Class distribution:")
for i, name in enumerate(CLASS_NAMES):
    cnt = (y == i).sum()
    pct = cnt / len(y) * 100
    print(f"      Grade {name}: {cnt} students ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] Train/test split (80/20, stratified) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"    Training samples: {len(X_train)}")
print(f"    Test samples    : {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Feature Scaling (for Logistic Regression only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Scaling features (StandardScaler for LR) ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("    Done.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Training models ...")

# --- Logistic Regression ---
print("    Training Logistic Regression ...")
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
t0 = time.time()
lr_model.fit(X_train_scaled, y_train)
lr_time = round(time.time() - t0, 4)
print(f"    Done in {lr_time}s")

# --- Random Forest ---
print("    Training Random Forest ...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
)
t0 = time.time()
rf_model.fit(X_train.values, y_train)
rf_time = round(time.time() - t0, 4)
print(f"    Done in {rf_time}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Evaluate Both Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Evaluating models ...")

y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))

def evaluate(name, model, X_te, train_time):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test_bin, y_prob, average="weighted", multi_class="ovr")
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()

    print(f"\n    ── {name} ──")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Train Time: {train_time}s")
    print(f"\n    Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))

    # ROC per class (one-vs-rest)
    roc_curves = {}
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        cls_auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))
        step = max(1, len(fpr) // 80)
        roc_curves[cls] = {
            "fpr": fpr[::step].tolist(),
            "tpr": tpr[::step].tolist(),
            "auc": round(cls_auc, 4),
        }

    return {
        "accuracy":       round(acc,     4),
        "precision":      round(prec,    4),
        "recall":         round(rec,     4),
        "f1_score":       round(f1,      4),
        "roc_auc":        round(roc_auc, 4),
        "train_time_sec": train_time,
        "confusion_matrix": cm,
        "class_names":    CLASS_NAMES,
    }, roc_curves

lr_metrics, lr_roc = evaluate("Logistic Regression", lr_model, X_test_scaled,  lr_time)
rf_metrics, rf_roc = evaluate("Random Forest",       rf_model, X_test.values,  rf_time)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
print("[7] Computing feature importance ...")

# Random Forest: native Gini importance
rf_imp = dict(zip(FEATURE_COLS, rf_model.feature_importances_.tolist()))

# Logistic Regression: mean |coefficient| across all classes
lr_coef_abs = np.mean(np.abs(lr_model.coef_), axis=0)
lr_imp = dict(zip(FEATURE_COLS, lr_coef_abs.tolist()))

feature_importance = {
    "Random Forest":       {k: round(v, 4) for k, v in rf_imp.items()},
    "Logistic Regression": {k: round(v, 4) for k, v in lr_imp.items()},
    "labels": FEATURE_LABELS,
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Dataset Summary
# ─────────────────────────────────────────────────────────────────────────────
class_counts = pd.Series(y).value_counts().sort_index()
dataset_summary = {
    "n_rows":          int(len(df)),
    "n_features":      int(len(FEATURE_COLS)),
    "class_counts":    {CLASS_NAMES[i]: int(class_counts.get(i, 0)) for i in range(len(CLASS_NAMES))},
    "feature_columns": FEATURE_COLS,
    "feature_labels":  FEATURE_LABELS,
    "train_size":      int(len(X_train)),
    "test_size":       int(len(X_test)),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save Everything
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[8] Saving artefacts to ./{MODELS_DIR}/ ...")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(lr_model,  f"{MODELS_DIR}/logistic_regression.pkl")
joblib.dump(rf_model,  f"{MODELS_DIR}/random_forest.pkl")
joblib.dump(scaler,    f"{MODELS_DIR}/scaler.pkl")
joblib.dump(imputer,   f"{MODELS_DIR}/imputer.pkl")

with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump({"Logistic Regression": lr_metrics, "Random Forest": rf_metrics}, f, indent=2)

with open(f"{MODELS_DIR}/feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=2)

with open(f"{MODELS_DIR}/roc_data.json", "w") as f:
    json.dump({"Logistic Regression": lr_roc, "Random Forest": rf_roc}, f, indent=2)

with open(f"{MODELS_DIR}/dataset_summary.json", "w") as f:
    json.dump(dataset_summary, f, indent=2)

print("\n" + "=" * 60)
print("  Training Complete! Files saved:")
for fname in os.listdir(MODELS_DIR):
    fpath = os.path.join(MODELS_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"    models/{fname}  ({size:,} bytes)")
print("=" * 60)
print("\n  Next steps:")
print("  Streamlit dashboard : streamlit run dashboard.py")
print("  Flask web app       : python app.py")
print("=" * 60)
