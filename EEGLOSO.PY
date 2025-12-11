
import os
from pathlib import Path
import gc

import numpy as np
import pandas as pd

import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils import shuffle
import joblib



def free_gpu_memory():
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ============================================================
# 0) GPU Random Forest (EEG-only ch1_/ch2_)
# ============================================================

print("Using cuML RandomForest on GPU (EEG-only ch1_/ch2_, 5-subject grouped CV LOSO).")


# ============================================================
# 1) Paths and basic settings  (WSL / Linux PATHS)
# ============================================================

DATA_PATH = Path("/mnt/c/Users/LENOVO/Downloads/FUSED_ALL_FINAL_FROM_DATA_ALL.csv")
MODELS_DIR = Path("/mnt/c/Users/LENOVO/Downloads/models_rf_5subj_loso_eeg_gpu")

SUBJECT_COL = "subject"
LABEL_COL   = "label"

RANDOM_STATE = 42
FOLD_SIZE_SUBJECTS = 5   

os.makedirs(MODELS_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


# ============================================================
# 2) Helper functions
# ============================================================

def undersample_multiclass(X, y, max_ratio=3.0, random_state=42):
    """Random undersampling for multi-class imbalance (CPU side)."""
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    max_per_class = {c: int(min_count * max_ratio) for c in classes}

    keep_idx = []

    for c in classes:
        idx = np.where(y == c)[0]
        n_keep = min(len(idx), max_per_class[c])
        chosen = rng.choice(idx, size=n_keep, replace=False)
        keep_idx.append(chosen)

    keep_idx = np.concatenate(keep_idx)
    keep_idx = shuffle(keep_idx, random_state=random_state)

    return X[keep_idx], y[keep_idx]


def get_gpu_rf(params):
    """Return cuML GPU RF model with given hyperparameters."""
    return cuRF(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],  # "sqrt"
        bootstrap=params["bootstrap"],
        random_state=RANDOM_STATE,
    )


def rf_fit_predict_gpu(X_train, y_train, X_test, params):
    """
    Train RF on GPU and predict labels.
    """
    rf = get_gpu_rf(params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    try:
        y_pred_np = cp.asnumpy(y_pred)
    except Exception:
        y_pred_np = np.asarray(y_pred)

    del y_pred
    gc.collect()
    free_gpu_memory()

    return rf, y_pred_np


def evaluate_metrics(y_true, y_pred, label_set=None):
    """Compute all metrics including confusion components."""
    if label_set is None:
        label_set = np.unique(y_true)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_set, zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=label_set)

    return {
        "labels": label_set,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cm": cm,
    }


def print_metrics(m, header=""):
    """Pretty-print metrics."""
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    labels = m["labels"]
    cm = m["cm"]

    print("\nPer-class metrics:")
    print("Label | Precision | Recall | F1 | Support")
    for i, c in enumerate(labels):
        print(f"{c:5d} | {m['precision'][i]:.4f} | {m['recall'][i]:.4f} | "
              f"{m['f1'][i]:.4f} | {m['support'][i]}")

    print("\nGlobal metrics:")
    print(f"Accuracy          : {m['accuracy']:.4f}")
    print(f"Balanced Accuracy : {m['balanced_accuracy']:.4f}")
    print(f"Micro-F1          : {m['micro_f1']:.4f}")
    print(f"Macro-F1          : {m['macro_f1']:.4f}")
    print(f"Weighted-F1       : {m['weighted_f1']:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    total = cm.sum()
    print("\nTP / FP / FN / TN per class:")
    for idx, c in enumerate(labels):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = total - (TP + FP + FN)
        print(f"Class {c}: TP={TP}, FP={FP}, FN={FN}, TN={TN}")


# ============================================================
# 3) Load dataset
# ============================================================

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)


# ============================================================
# 4) Ensure only NUMERIC EEG features (ch1_/ch2_) + subject/label
# ============================================================

print("\nFiltering numeric EEG columns ch1_/ch2_ only...")

df_numeric = df.select_dtypes(include=[np.number]).copy()

meta_cols = [SUBJECT_COL, LABEL_COL]
meta_cols = [c for c in meta_cols if c in df_numeric.columns]

all_feature_cols = [c for c in df_numeric.columns if c not in meta_cols]

feature_cols = [
    c for c in all_feature_cols
    if c.lower().startswith("ch1_") or c.lower().startswith("ch2_")
]

if len(feature_cols) == 0:
    raise ValueError(
        "No EEG columns starting with ch1_ or ch2_ were found. "
        "Check your CSV column names."
    )

print(f"Numeric EEG feature columns (ch1_/ch2_): {len(feature_cols)}")

X_all = df_numeric[feature_cols].values.astype(np.float32)
y_all = df[LABEL_COL].values.astype(np.int32)
subjects_all = df[SUBJECT_COL].values
label_set = np.unique(y_all)


# ============================================================
# 5) Fixed hyperparameters for GPU RF
# ============================================================

FIXED_PARAMS = {
    "n_estimators": 400,
    "max_depth": 24,
    "max_features": "sqrt",
    "bootstrap": True,
}

print("\nUsing fixed RF hyperparameters (no nested tuning):")
print(FIXED_PARAMS)


# ============================================================
# 6) 5-subject grouped CV LOSO 
# ============================================================

unique_subjects = np.sort(np.unique(subjects_all))
print("\n===== 5-subject grouped CV LOSO (EEG GPU) =====")
print("Total subjects:", len(unique_subjects))

all_true = []
all_pred = []
outer_f1 = []

cv_rows = []
fold_idx = 1

for start in range(0, len(unique_subjects), FOLD_SIZE_SUBJECTS):
    test_subs = unique_subjects[start:start + FOLD_SIZE_SUBJECTS]

    print("\n" + "#" * 80)
    print(f"Fold {fold_idx} - Test subjects: {test_subs}")
    print("#" * 80)

    is_test = np.isin(subjects_all, test_subs)
    is_train = ~is_test

    X_tr = X_all[is_train]
    y_tr = y_all[is_train]
    X_te = X_all[is_test]
    y_te = y_all[is_test]

    if X_te.shape[0] == 0:
        print(f"⚠ Fold {fold_idx}: no test rows, skipping.")
        fold_idx += 1
        continue

    print("  Train rows:", X_tr.shape[0])
    print("  Test  rows:", X_te.shape[0])

    # undersample (CPU side → numpy)
    X_tr_bal, y_tr_bal = undersample_multiclass(
        X_tr, y_tr, max_ratio=3.0,
        random_state=RANDOM_STATE + fold_idx
    )
    print("  Train rows after undersampling:", X_tr_bal.shape[0])

    # Train on GPU + predict labels
    model, y_pred = rf_fit_predict_gpu(
        X_tr_bal, y_tr_bal, X_te, FIXED_PARAMS
    )

    model_path = MODELS_DIR / f"rf_fold_{fold_idx:02d}_eeg_gpu_5subj.pkl"
    joblib.dump(model, model_path)
    print("  Saved fold model to:", model_path)

    # evaluation
    m = evaluate_metrics(y_te, y_pred, label_set)
    print_metrics(m, header=f"Fold {fold_idx} Test Metrics (EEG GPU, 5-subject CV)")

    all_true.append(y_te)
    all_pred.append(y_pred)
    outer_f1.append(m["macro_f1"])

    cv_rows.append({
        "fold": fold_idx,
        "test_subjects": ",".join(str(s) for s in test_subs),
        "n_train": int(X_tr_bal.shape[0]),
        "n_test": int(X_te.shape[0]),
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "micro_f1": m["micro_f1"],
    })

    del X_tr, y_tr, X_te, y_te, X_tr_bal, y_tr_bal, model, y_pred, m
    gc.collect()
    free_gpu_memory()

    fold_idx += 1


# ============================================================
# 7) GLOBAL METRICS
# ============================================================

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

global_m = evaluate_metrics(all_true, all_pred, label_set)
print_metrics(global_m, header="GLOBAL Metrics Across All 5-subject CV Folds (EEG GPU)")


# ============================================================
# 8) Save CV summary to CSV
# ============================================================

cv_df = pd.DataFrame(cv_rows)
cv_path = MODELS_DIR / "rf_eeg_gpu_5subj_cv_summary.csv"
cv_df.to_csv(cv_path, index=False)

print("\nSaved 5-subject CV summary to:", cv_path)
print("\n5-subject CV summary (head):")
print(cv_df.head())


# ============================================================
# 9) FINAL DEPLOYMENT MODEL (GPU) FIXED_PARAMS
# ============================================================

# undersample full data (CPU side)
X_full_bal, y_full_bal = undersample_multiclass(
    X_all, y_all, max_ratio=3.0, random_state=999
)

final_model, _ = rf_fit_predict_gpu(
    X_full_bal, y_full_bal, X_full_bal, FIXED_PARAMS
)

final_path = MODELS_DIR / "rf_final_deployment_eeg_gpu_5subj.pkl"
joblib.dump(final_model, final_path)

print("\nFinal deployment model saved to:", final_path)
print("Use this model for real-time predictions (EEG-only, GPU, 5-subject CV).")
