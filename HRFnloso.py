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
import joblib
from sklearn.impute import SimpleImputer


from imblearn.over_sampling import SMOTE  # Balanced SMOTE

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
# 0) GPU Random Forest (ALL numeric features)
# ============================================================

print("Using cuML RandomForest on GPU (ALL numeric features + Hybrid Balanced SMOTE).")


# ============================================================
# 1) Paths and basic settings  (WSL / Linux PATHS)
# ============================================================

DATA_PATH = Path("/mnt/c/Users/LENOVO/Downloads/Senior_Proj/Final/Data/FUSED_ALL_FINAL_FROM_DATA_ALL.csv")

# New models directory name (ALL features + Hybrid SMOTE)
MODELS_DIR = Path("/mnt/c/Users/LENOVO/Downloads/models_nested_rf_allfeat_hybridsmote_gpu")

SUBJECT_COL = "subject"
LABEL_COL   = "label"

RANDOM_STATE = 42
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 10

os.makedirs(MODELS_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


# ============================================================
# 2) Helper functions
# ============================================================

def make_subject_folds(subject_ids, n_folds=10, random_state=42):
    """Split subjects into n folds (subject-wise)."""
    unique_subjects = np.array(sorted(np.unique(subject_ids)))
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_subjects)
    folds = np.array_split(unique_subjects, n_folds)
    return folds


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


def hybrid_resample_balanced_smote(X, y, max_ratio=3.0, random_state=42):
    """
    Hybrid resampling:
    1) Random undersampling for majority.
    2) Balanced SMOTE (oversample non-majority classes).
    """
    # Step 1: undersampling
    X_under, y_under = undersample_multiclass(
        X, y, max_ratio=max_ratio, random_state=random_state
    )

    # Step 2: Balanced SMOTE
    try:
        sm = SMOTE(
            sampling_strategy="not majority",  # oversample labels 1 و 2
            random_state=random_state
        )
        X_res, y_res = sm.fit_resample(X_under, y_under)
        X_res = X_res.astype(np.float32)
        y_res = y_res.astype(np.int32)
        return X_res, y_res
    except Exception as e:
        print(f"[WARN] SMOTE failed, using undersampling only. Reason: {e}")
        X_under = X_under.astype(np.float32)
        y_under = y_under.astype(np.int32)
        return X_under, y_under


def get_gpu_rf(params):
    """Return cuML GPU RF model with given hyperparameters."""
    return cuRF(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],  # "sqrt"
        bootstrap=params["bootstrap"],
        random_state=RANDOM_STATE,
    )


def rf_fit_predict_proba_gpu(X_train, y_train, X_val, params):
    """
    Train RF on GPU and predict probabilities.
    Returns numpy arrays only (no model object).
    Used in the inner loop (hyperparameter tuning).
    """
    rf = get_gpu_rf(params)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_val)

    # Convert to numpy from cupy if needed
    try:
        y_proba_np = cp.asnumpy(y_proba)
    except Exception:
        y_proba_np = np.asarray(y_proba)

    # Cleanup
    del rf, y_proba
    gc.collect()
    free_gpu_memory()

    return y_proba_np


def rf_fit_predict_gpu(X_train, y_train, X_test, params):
    """
    Train RF on GPU and predict labels.
    Returns the model object and numpy predictions.
    """
    rf = get_gpu_rf(params)
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


def compute_inner_auc(
    X,
    y,
    subjects,
    label_set,
    param_grid,
    n_inner_folds=10,
    max_ratio=3.0,
    outer_idx=None,
):
    """
    Inner-loop ROC AUC tuning (GPU RF + Hybrid Balanced SMOTE).
    Adds progress logs: OUTER / COMBO / INNER.
    """
    inner_folds = make_subject_folds(
        subjects, n_folds=n_inner_folds, random_state=RANDOM_STATE
    )
    combo_scores = []

    print("\n------------------------------------------------------")
    print("Inner loop ROC AUC hyperparameter tuning (GPU + Hybrid SMOTE)")
    if outer_idx is not None:
        print(f"(For OUTER fold {outer_idx})")
    print("------------------------------------------------------")

    total_combos = len(param_grid)

    for idx, params in enumerate(param_grid):
        fold_aucs = []
        combo_num = idx + 1

        print(f"\n[OUTER {outer_idx} | COMBO {combo_num}/{total_combos}] Starting inner CV...")

        for inner_idx, val_subj in enumerate(inner_folds):
            is_val = np.isin(subjects, val_subj)
            is_train = ~is_val

            X_tr = X[is_train]
            y_tr = y[is_train]
            X_val = X[is_val]
            y_val = y[is_val]

            # Hybrid undersampling + Balanced SMOTE (CPU → numpy)
            X_tr_bal, y_tr_bal = hybrid_resample_balanced_smote(
                X_tr,
                y_tr,
                max_ratio=max_ratio,
                random_state=RANDOM_STATE + inner_idx,
            )

            # Train RF on GPU + predict probabilities
            y_proba = rf_fit_predict_proba_gpu(
                X_tr_bal, y_tr_bal, X_val, params
            )

            try:
                auc = roc_auc_score(
                    y_val,
                    y_proba,
                    labels=label_set,
                    multi_class="ovr",
                    average="macro",
                )
                fold_aucs.append(auc)
            except Exception:
                pass

            print(
                f"   [OUTER {outer_idx} | COMBO {combo_num}/{total_combos} | "
                f"INNER {inner_idx+1}/{len(inner_folds)}] done"
            )

            # تنظيف
            del X_tr, y_tr, X_val, y_val, X_tr_bal, y_tr_bal, y_proba
            gc.collect()
            free_gpu_memory()

        mean_auc = np.mean(fold_aucs) if len(fold_aucs) > 0 else np.nan
        combo_scores.append((params, mean_auc))
        print(f"Combo {combo_num}/{total_combos} {params} → mean ROC AUC = {mean_auc:.4f}")

        del fold_aucs
        gc.collect()
        free_gpu_memory()

    valid = [c for c in combo_scores if not np.isnan(c[1])]
    if len(valid) == 0:
        raise RuntimeError("No valid AUC scores in inner loop (all NaN).")

    best_params, best_auc = max(valid, key=lambda x: x[1])

    print(f"\n[OUTER {outer_idx}] Best inner params: {best_params} with AUC={best_auc:.4f}")

    return best_params, best_auc



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

imputer = SimpleImputer(strategy="median")
X_all = imputer.fit_transform(df_numeric[feature_cols]).astype(np.float32)

y_all = df[LABEL_COL].values.astype(np.int32)
subjects_all = df[SUBJECT_COL].values
label_set = np.unique(y_all)


# ============================================================
# 5) Hyperparameter grid  (9 combinations)
# ============================================================

param_grid = []
for ne in [300, 400, 500]:      # 3 values
    for md in [16, 24, 32]:     # 3 values
        param_grid.append({
            "n_estimators": ne,
            "max_depth": md,
            "max_features": "sqrt",  # 3×3×1 = 9
            "bootstrap": True,
        })

print("\nTotal hyperparameter combinations:", len(param_grid))


# ============================================================
# 6) Outer Loop (10 folds, subject-wise LOSO-style)
# ============================================================

outer_folds = make_subject_folds(
    subjects_all, n_folds=N_OUTER_FOLDS, random_state=RANDOM_STATE
)

all_true = []
all_pred = []

outer_best_params = []
outer_best_auc = []
outer_f1 = []

for outer_idx, test_subj in enumerate(outer_folds, start=1):

    print("\n" + "#" * 80)
    print(f"OUTER Fold {outer_idx}/10")
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

    # -------------------------
    # INNER LOOP (AUC tuning, GPU + Hybrid SMOTE)
    # -------------------------
    best_params, best_auc = compute_inner_auc(
        X_trval,
        y_trval,
        subj_trval,
        label_set,
        param_grid,
        n_inner_folds=N_INNER_FOLDS,
        max_ratio=3.0,
        outer_idx=outer_idx,  
    )

    outer_best_params.append(best_params)
    outer_best_auc.append(best_auc)

    # -------------------------
    # Train FINAL model for this outer fold (GPU + Hybrid SMOTE)
    # -------------------------
    X_trval_bal, y_trval_bal = hybrid_resample_balanced_smote(
        X_trval, y_trval, max_ratio=3.0,
        random_state=RANDOM_STATE + outer_idx
    )

    model, y_pred = rf_fit_predict_gpu(
        X_trval_bal, y_trval_bal, X_test, best_params
    )

    # New model filename (ALL features + Hybrid SMOTE)
    model_path = MODELS_DIR / f"rf_outer_{outer_idx:02d}_allfeat_hybridsmote_gpu.pkl"
    joblib.dump(model, model_path)
    print("Saved model:", model_path)

    # evaluation
    m = evaluate_metrics(y_test, y_pred, label_set)
    print_metrics(m, header=f"OUTER Fold {outer_idx} Test Metrics (ALL features, Hybrid SMOTE, GPU)")

    all_true.append(y_test)
    all_pred.append(y_pred)
    outer_f1.append(m["macro_f1"])

    # Cleanup after each outer fold
    del X_trval, y_trval, subj_trval, X_test, y_test
    del X_trval_bal, y_trval_bal, model, y_pred, m
    gc.collect()
    free_gpu_memory()


# ============================================================
# 7) GLOBAL METRICS
# ============================================================

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

global_m = evaluate_metrics(all_true, all_pred, label_set)
print_metrics(global_m, header="GLOBAL Metrics Across All 10 OUTER Folds (ALL features, Hybrid SMOTE, GPU)")


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

# New AUC summary filename
auc_path = MODELS_DIR / "inner_auc_summary_allfeat_hybridsmote_gpu.csv"
auc_df.to_csv(auc_path, index=False)

print("\nSaved inner AUC summary to:", auc_path)


# ============================================================
# 9) FINAL DEPLOYMENT MODEL (GPU + Hybrid SMOTE)
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

print("\nSelected FINAL deployment hyperparameters (ALL features, Hybrid SMOTE, GPU):", final_params)

# Hybrid resampling on full dataset
X_full_bal, y_full_bal = hybrid_resample_balanced_smote(
    X_all, y_all, max_ratio=3.0, random_state=999
)

final_model, _ = rf_fit_predict_gpu(
    X_full_bal, y_full_bal, X_full_bal, final_params
)

# New final model filename
final_path = MODELS_DIR / "rf_final_deployment_allfeat_hybridsmote_gpu.pkl"
joblib.dump(final_model, final_path)

print("\nFinal deployment model saved to:", final_path)
print("Use this model for real-time predictions (ALL numeric features, Hybrid SMOTE, GPU).")