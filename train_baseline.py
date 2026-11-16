"""Step 4: baseline model = Morgan fingerprint + random forest.

Reads:  data/er_split.csv, data/ar_split.csv, data/er_fingerprints.npy, data/ar_fingerprints.npy
Writes: data/er_baseline_scores.csv, data/ar_baseline_scores.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             fbeta_score, matthews_corrcoef)

LABEL_COL = "label"
SPLIT_COLS = {"random": "random_split", "scaffold": "scaffold_split"}
# split columns hold "train", "val" or "test" (as written by split_data.py)
SEED = 42
DATA_DIR = "data"


def best_f2_threshold(y_true, scores):
    """Pick the threshold that maximizes F2 on the VALIDATION set only."""
    best_t, best_f2 = 0.5, -1
    for t in np.unique(np.round(scores, 4)):
        f2 = fbeta_score(y_true, scores >= t, beta=2, zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = t, f2
    return best_t


def evaluate(name, y_val, s_val, y_test, s_test):
    t = best_f2_threshold(y_val, s_val)
    pred = s_test >= t
    print(f"    {name:<8} PR-AUC {average_precision_score(y_test, s_test):.3f} | "
          f"ROC-AUC {roc_auc_score(y_test, s_test):.3f} | "
          f"F2 {fbeta_score(y_test, pred, beta=2, zero_division=0):.3f} | "
          f"MCC {matthews_corrcoef(y_test, pred):.3f} | "
          f"no-skill PR-AUC {y_test.mean():.3f}")


for receptor in ["er", "ar"]:
    df = pd.read_csv(f"{DATA_DIR}/{receptor}_split.csv")
    X = np.load(f"{DATA_DIR}/{receptor}_fingerprints.npy")
    y = df[LABEL_COL].to_numpy()
    out = df.copy()

    for split_name, col in SPLIT_COLS.items():
        tr, va, te = (df[col] == p for p in ["train", "val", "test"])
        print(f"{receptor.upper()} {split_name}: "
              f"{tr.sum()} train / {va.sum()} val / {te.sum()} test "
              f"({y[te].sum()} active in test)")

        # Main baseline: random forest on the fingerprint
        rf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)
        rf.fit(X[tr], y[tr])
        s = rf.predict_proba(X)[:, 1]
        out[f"rf_score_{split_name}"] = s
        evaluate("RF", y[va], s[va], y[te], s[te])

        # Control: same classifier as Step 6 (logistic regression), fingerprint input.
        # Lets you separate "better input" from "better classifier".
        scaler = StandardScaler().fit(X[tr])
        lr = LogisticRegression(class_weight="balanced", max_iter=2000)
        lr.fit(scaler.transform(X[tr]), y[tr])
        s_lr = lr.predict_proba(scaler.transform(X))[:, 1]
        out[f"logreg_score_{split_name}"] = s_lr
        evaluate("LogReg", y[va], s_lr[va], y[te], s_lr[te])

    out.to_csv(f"{DATA_DIR}/{receptor}_baseline_scores.csv", index=False)
