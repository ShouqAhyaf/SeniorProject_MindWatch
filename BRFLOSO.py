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
    roc_auc_score
)
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
import joblib

# ============================================================
# Small helper to free GPU memory between trainings
# ============================================================

def free_gpu_memory():
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ============================================================
# 0) GPU Random Forest (ALL numeric features, undersample + class weights)
#    5-subject grouped CV + sample_weight 

print("Using cuML RandomForest on GPU (ALL numeric features + undersampling + class weights, 5-subject grouped CV).")


# ============================================================
# 1) Paths and basic settings  (WSL / Linux PATHS)
# ============================================================

DATA_PATH = Path("/mnt/c/Users/LENOVO/Downloads/FUSED_ALL_FINAL_FROM_DATA_ALL.csv")

MODELS_DIR = Path("/mnt/c/Users/LENOVO/Downloads/models_rf_5subj_allfeat_under_weight_gpu_sampleweight")

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
    """
    Random undersampling for multi-class imbalance (CPU side).

    Keeps each class size <= max_ratio * minority_count.
    """
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


def compute_class_weights(y):
    """
    Compute class weights using:
        weight_c = total / (n_classes * count_c)
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)

    class_weights = {}
    for c, cnt in zip(classes, counts):
        w_c = total / (n_classes * cnt)
        class_weights[c] = float(w_c)
    return class_weights


def build_sample_weights(y, class_weights):
    """
    Build per-sample weight vector from class_weights dict.
    """
    y = np.asarray(y)
    w = np.zeros_like(y, dtype=np.float32)
    for c, w_c in class_weights.items():
        w[y == c] = np.float32(w_c)
    return w


def get_gpu_rf(params):
    """Return cuML GPU RF model with given hyperparameters."""
    return cuRF(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],  # "sqrt"
        bootstrap=params["bootstrap"],
        random_state=RANDOM_STATE,
    )


def rf_fit_predict_gpu(X_train, y_train, X_test, params, sample_weight=None):
    """
    Train RF on GPU and predict labels.
    Returns the model object and numpy predictions.
    """
    rf = get_gpu_rf(params)
    if sample_weight is not None:
        rf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    try:
        y_pred_np = cp.asnumpy(y_pred)
    except Exception:
        y_pred_np = np.asarray(y_pred)

    # Cleanup buffers
    del y_pred
    gc.collect()
    free_gpu_memory()

    return rf, y_pred_np


def evaluate_metrics(y_true, y_pred, label_set=None):
    """Compute all metrics including confusion matrix components."""
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
# 4) Use ALL numeric features + subject/label
# ============================================================
print("\nSelecting ALL numeric feature columns (not only EEG)...")

df_numeric = df.select_dtypes(include=[np.number]).copy()

meta_cols = [SUBJECT_COL, LABEL_COL]
meta_cols = [c for c in meta_cols if c in df_numeric.columns]

# All numeric columns except meta (subject, label)
feature_cols = [c for c in df_numeric.columns if c not in meta_cols]

if len(feature_cols) == 0:
    raise ValueError(
        "No numeric feature columns found (excluding subject/label). "
        "Check your CSV column names and dtypes."
    )

print(f"Total numeric feature columns (ALL): {len(feature_cols)}")

# Impute NaNs once globally
imputer = SimpleImputer(strategy="median")
X_all = imputer.fit_transform(df_numeric[feature_cols]).astype(np.float32)

y_all = df[LABEL_COL].values.astype(np.int32)
subjects_all = df[SUBJECT_COL].values
label_set = np.unique(y_all)


# ============================================================
# 5) FIXED hyperparameters
# ============================================================

FIXED_PARAMS = {
    "n_estimators": 400,
    "max_depth": 24,
    "max_features": "sqrt",
    "bootstrap": True,
}




unique_subjects = np.array(sorted(np.unique(subjects_all)))
outer_folds = [
    unique_subjects[i:i + FOLD_SIZE_SUBJECTS]
    for i in range(0, len(unique_subjects), FOLD_SIZE_SUBJECTS)
]

all_true = []
all_pred = []

outer_best_params = []
outer_best_auc = []   # placeholder 
outer_f1 = []

print(f"\nTotal subjects: {len(unique_subjects)}, "
      f"Number of folds (5-subject CV): {len(outer_folds)}")

for outer_idx, test_subj in enumerate(outer_folds, start=1):

    print("\n" + "#" * 80)
    print(f"OUTER Fold {outer_idx}/{len(outer_folds)} (5-subject grouped CV)")
    print("#" * 80)

    is_test = np.isin(subjects_all, test_subj)
    is_trainval = ~is_test

    X_trval = X_all[is_trainval]
    y_trval = y_all[is_trainval]
    subj_trval = subjects_all[is_trainval]

    X_test = X_all[is_test]
    y_test = y_all[is_test]

    print("Train+Val subjects:", len(np.unique(subj_trval)))
    print("Test subjects:", len(np.unique(test_subj)))

    best_params = FIXED_PARAMS.copy()
    best_auc = np.nan
    outer_best_params.append(best_params)
    outer_best_auc.append(best_auc)

    # -------------------------
    # Train FINAL model for this outer fold (GPU + undersample + class weights via sample_weight)
    # -------------------------
    # 1) Undersampling
    X_trval_bal, y_trval_bal = undersample_multiclass(
        X_trval, y_trval, max_ratio=3.0,
        random_state=RANDOM_STATE + outer_idx
    )

    # 2) Class weights
    class_weights_outer = compute_class_weights(y_trval_bal)
    sample_w_outer = build_sample_weights(y_trval_bal, class_weights_outer)

    print("  Train rows after undersampling:", X_trval_bal.shape[0])

    model, y_pred = rf_fit_predict_gpu(
        X_trval_bal, y_trval_bal, X_test, best_params, sample_weight=sample_w_outer
    )

    # model filename
    model_path = MODELS_DIR / f"rf_outer_{outer_idx:02d}_allfeat_under_weight_gpu_5subj_sw.pkl"
    joblib.dump(model, model_path)
    print("Saved model:", model_path)

    # evaluation
    m = evaluate_metrics(y_test, y_pred, label_set)
    print_metrics(
        m,
        header=f"OUTER Fold {outer_idx} Test Metrics (ALL features, undersample + class weights(sample_weight), GPU, 5-subject CV)"
    )

    all_true.append(y_test)
    all_pred.append(y_pred)
    outer_f1.append(m["macro_f1"])

    # Cleanup after each outer fold
    del X_trval, y_trval, subj_trval, X_test, y_test
    del X_trval_bal, y_trval_bal
    del model, y_pred, m, class_weights_outer, sample_w_outer
    gc.collect()
    free_gpu_memory()


# ============================================================
# 7) GLOBAL METRICS
# ============================================================

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

global_m = evaluate_metrics(all_true, all_pred, label_set)
print_metrics(
    global_m,
    header="GLOBAL Metrics Across All 5-subject CV Folds (ALL features, undersample + class weights(sample_weight), GPU)"
)


# ============================================================
# 8) Save AUC summary 
# ============================================================

rows = []
for i, (p, auc_val) in enumerate(zip(outer_best_params, outer_best_auc), start=1):
    rows.append({
        "outer_fold": i,
        "best_inner_mean_roc_auc": auc_val,
        "n_estimators": p["n_estimators"],
        "max_depth": p["max_depth"],
        "max_features": p["max_features"],
        "bootstrap": p["bootstrap"],
    })

auc_df = pd.DataFrame(rows)

auc_path = MODELS_DIR / "inner_auc_summary_allfeat_under_weight_gpu_5subj_sw.csv"
auc_df.to_csv(auc_path, index=False)

print("\nSaved (dummy) inner AUC summary to:", auc_path)


# ============================================================
# 9) FINAL DEPLOYMENT MODEL (GPU + undersample + class weights(sample_weight))
# ============================================================

from collections import defaultdict

count = defaultdict(int)
sum_f1 = defaultdict(float)

for params, f1_val in zip(outer_best_params, outer_f1):
    key = tuple(sorted(params.items()))
    count[key] += 1
    sum_f1[key] += float(f1_val)

best_key = None
best_count = -1
best_f1sum = -np.inf

for k in count:
    if count[k] > best_count or (count[k] == best_count and sum_f1[k] > best_f1sum):
        best_key = k
        best_count = count[k]
        best_f1sum = sum_f1[k]

final_params = dict(best_key)

print("\nSelected FINAL deployment hyperparameters (ALL features, undersample + class weights(sample_weight), GPU, 5-subject CV):", final_params)

# Undersample on full dataset
X_full_bal, y_full_bal = undersample_multiclass(
    X_all, y_all, max_ratio=3.0, random_state=999
)

class_weights_full = compute_class_weights(y_full_bal)
sample_w_full = build_sample_weights(y_full_bal, class_weights_full)

final_model, _ = rf_fit_predict_gpu(
    X_full_bal, y_full_bal, X_full_bal, final_params, sample_weight=sample_w_full
)

final_path = MODELS_DIR / "rf_final_deployment_allfeat_under_weight_gpu_5subj_sw.pkl"
joblib.dump(final_model, final_path)

print("\nFinal deployment model saved to:", final_path)
print("Use this model for real-time predictions (ALL numeric features, undersample + class weights(sample_weight), GPU, 5-subject CV).")
