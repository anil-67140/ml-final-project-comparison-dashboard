# Student Performance Prediction — ML Course Project

Predicts a student's **GradeClass** (A / B / C / D / F) using **Logistic Regression**
and **Random Forest**, with a Flask website and a Streamlit dashboard. Everything
here is **plain Python source code** — nothing pre-trained is shipped. You run the
training script yourself and it generates the model files locally.

---

## What generated what (for your instructor)

| File | What it does |
|---|---|
| `train_models.py` | **This is the file that generates everything else.** It loads the CSV, cleans it, trains Logistic Regression + Random Forest, evaluates them, and saves the trained models + all metrics/charts data into a `models/` folder it creates. |
| `app.py` | Flask website. Loads the files created by `train_models.py` and serves two pages: a prediction form and a comparison dashboard. |
| `dashboard.py` | Streamlit dashboard. Same idea as `app.py` but built with Streamlit instead of raw HTML. |
| `Students_Performance_Dataset.csv` | The dataset (Kaggle). |

**You must run `train_models.py` first.** It is the only file that touches scikit-learn
training code — `app.py` and `dashboard.py` only *load* what it produced and display it.
If your instructor asks "where did these files come from?" — point to `train_models.py`
and show them it runs from a plain CSV with no shortcuts.

---

## Folder structure

```
ml_project/
├── Students_Performance_Dataset.csv   ← dataset
├── train_models.py                    ← RUN THIS FIRST
├── app.py                             ← Flask website
├── dashboard.py                       ← Streamlit dashboard
├── requirements.txt
└── models/                            ← created automatically by train_models.py
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── metrics.json
    ├── feature_importance.json
    ├── roc_data.json
    └── dataset_summary.json
```

---

## How to run (step by step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models — generates everything
```bash
python train_models.py
```
You will see full console output: dataset shape, missing values, class
distribution, training progress, accuracy/precision/recall/F1/ROC-AUC for
both models, and a full classification report — then it saves the `models/`
folder.

### 3a. Run the Flask website
```bash
python app.py
```
Open **http://127.0.0.1:5000**

- `/` — Enter a student's details in the form → both models predict the
  grade at the same time, side-by-side, with probability bars and an
  agree/disagree banner.
- `/compare` — Full comparison dashboard: metrics table, bar charts,
  confusion matrices, ROC curves, feature importance, dataset stats.
- `/api/predict` (POST, JSON) — programmatic prediction endpoint.

### 3b. OR run the Streamlit dashboard
```bash
streamlit run dashboard.py
```
Open **http://localhost:8501**

Sidebar lets you enter a student profile; 6 tabs cover live prediction,
metrics comparison, confusion matrices, ROC curves, feature importance,
and dataset overview.

You can run **both** at the same time on different ports if you want to
show your instructor both versions.

---

## Dataset

**Students Performance Dataset** — Rabie El Kharoua, Kaggle, CC BY 4.0
https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

2,392 students, 12 input features + GPA + GradeClass target.

> Note: the Kaggle download endpoint needs API credentials that aren't
> available in some sandboxed environments. If you want the *exact* file
> from Kaggle, log into Kaggle yourself and download it directly — it will
> drop into `train_models.py` with no code changes needed, since the
> column names match the dataset's published schema exactly.

---

## Important: why GPA is excluded

`GradeClass` is **computed directly from GPA** (GPA ≥ 3.5 → A, 3.0–3.5 → B,
etc.). If GPA were left in as a feature, both models would score ~100%
accuracy without learning anything — that's data leakage, not a good
result. `train_models.py` deliberately drops GPA (and StudentID, which is
just an identifier) before training. This is explained in the code
comments and printed to the console when you run the script.

---

## Expected results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | ~49% | ~48% |
| Precision | ~38% | ~36% |
| Recall | ~49% | ~48% |
| F1-Score | ~38% | ~35% |
| ROC-AUC | ~0.675 | ~0.669 |
| Train Time | ~0.01s | ~0.6s |

Moderate accuracy (~45–50%) on this 5-class problem is expected and
correct — it's a genuinely hard task once GPA is removed to prevent
leakage, and adjacent grade bands (like B vs C) overlap heavily in the
feature space.
