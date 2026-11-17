"""Evaluate saved frozen and fine-tuned ChemBERTa predictions.

Run from the project root after training:
    python evaluate_metrics.py

Thresholds are chosen on validation data to maximize F2, then locked for
unseen test data. PR-AUC here is sklearn average precision (AP), matching the
existing train_frozen.py and train_finetune.py scripts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef

DATA_DIR = Path("data")
THRESHOLDS = np.linspace(0.01, 0.99, 99)
results = []

for assay in ("er", "ar"):
    for model in ("frozen", "finetuned"):
        file = DATA_DIR / f"{assay}_{model}_scores.csv"
        if not file.exists():
            print(f"SKIPPING {file}: run the corresponding training script first")
            continue
        df = pd.read_csv(file)

        for split in ("random", "scaffold"):
            split_col = f"{split}_split"
            score_col = f"{model}_{split}_score"
            if split_col not in df or score_col not in df:
                print(f"SKIPPING {file}: missing {split_col} or {score_col}")
                continue

            validation = df.loc[df[split_col] == "val", ["label", score_col]].dropna()
            test = df.loc[df[split_col] == "test", ["label", score_col]].dropna()
            if validation.empty or test.empty:
                print(f"SKIPPING {assay}/{model}/{split}: validation or test predictions missing")
                continue

            val_true = validation["label"].to_numpy(dtype=int)
            val_scores = validation[score_col].to_numpy(dtype=float)
            test_true = test["label"].to_numpy(dtype=int)
            test_scores = test[score_col].to_numpy(dtype=float)

            # Select threshold using ONLY validation data. In ties, choose the
            # first (lowest) threshold for a deterministic, recall-friendly rule.
            val_f2 = [fbeta_score(val_true, val_scores >= t, beta=2, zero_division=0)
                      for t in THRESHOLDS]
            best_threshold = float(THRESHOLDS[int(np.argmax(val_f2))])
            test_pred = (test_scores >= best_threshold).astype(int)

            # Average precision is defined for binary classes but is not
            # informative if the test partition contains just one class.
            pr_auc = (average_precision_score(test_true, test_scores)
                      if np.unique(test_true).size == 2 else float("nan"))

            results.append({
                "assay": assay.upper(),
                "model": model,
                "split": split,
                "n_test": len(test),
                "test_positive_rate": float(test_true.mean()),
                "validation_F2_threshold": best_threshold,
                "F2": fbeta_score(test_true, test_pred, beta=2, zero_division=0),
                "MCC": matthews_corrcoef(test_true, test_pred),
                "PR_AUC_average_precision": pr_auc,
            })

if not results:
    raise SystemExit("No saved prediction files found in data/. Train models first.")

summary = pd.DataFrame(results)
DATA_DIR.mkdir(exist_ok=True)
output = DATA_DIR / "evaluation_metrics.csv"
summary.to_csv(output, index=False)
print("\nTEST RESULTS (validation-selected F2 thresholds):")
print(summary.round(4).to_string(index=False))
print(f"\nSaved {output}")
