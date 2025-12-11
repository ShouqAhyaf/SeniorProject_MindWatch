import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
import joblib
import cupy as cp

DATA_PATH = Path("/mnt/c/Users/LENOVO/Downloads/FUSED_ALL_FINAL_FROM_DATA_ALL.csv")
MODELS_DIR = Path("/mnt/c/Users/LENOVO/Downloads")
MODEL_PATH = MODELS_DIR / "rf_final_deployment.pkl"

SUBJECT_COL = "subject"
LABEL_COL = "label"

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df_numeric = df.select_dtypes(include=[np.number]).copy()

meta_cols = [c for c in [SUBJECT_COL, LABEL_COL] if c in df_numeric.columns]
feature_cols = [c for c in df_numeric.columns if c not in meta_cols]

imputer = SimpleImputer(strategy="median")
X_all = imputer.fit_transform(df_numeric[feature_cols]).astype(np.float32)
y_all = df[LABEL_COL].values.astype(np.int32)

wanted_labels = [0, 1, 2]
chosen_indices = []

for lbl in wanted_labels:
    idxs = np.where(y_all == lbl)[0]
    if len(idxs) > 0:
        chosen_indices.append(int(idxs[0]))

X_demo = X_all[chosen_indices]
y_demo_true = y_all[chosen_indices]

y_demo_pred = model.predict(X_demo)
if isinstance(y_demo_pred, cp.ndarray):
    y_demo_pred = cp.asnumpy(y_demo_pred)
else:
    y_demo_pred = np.asarray(y_demo_pred)

print("\nIndex | True | Pred")
print("---------------------")
for idx_csv, t, p in zip(chosen_indices, y_demo_true, y_demo_pred):
    print(f"{idx_csv} | {t} | {p}")


def predict_from_dataframe(df_new):
    X_new = df_new[feature_cols].copy()
    X_new_imputed = imputer.transform(X_new).astype(np.float32)
    y_new_pred = model.predict(X_new_imputed)
    if isinstance(y_new_pred, cp.ndarray):
        y_new_pred = cp.asnumpy(y_new_pred)
    else:
        y_new_pred = np.asarray(y_new_pred)
    return y_new_pred


example_rows = df.iloc[100:103].copy()
y_example_true = example_rows[LABEL_COL].values.astype(int)
y_example_pred = predict_from_dataframe(example_rows)

print("\nExample rows 100–102:")
for i in range(len(example_rows)):
    print(f"Row {100+i}: true={y_example_true[i]} pred={int(y_example_pred[i])}")
