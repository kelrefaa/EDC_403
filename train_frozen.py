# train_frozen.py - Step 7: train a simple classifier on the frozen ChemBERTa embeddings.
#
# What it does: same plan as train_baseline.py, but the input is each chemical's 384-number ChemBERTa
# embedding instead of its Morgan fingerprint, and the classifier is logistic regression instead of a forest.
# It prints the same two scores on the test chemicals, plus training time, and saves every score.
#
# Run it with:  python train_frozen.py
# It creates:   data/er_frozen_scores.csv  and  data/ar_frozen_scores.csv

import time

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import PrecisionRecallDisplay

# For visualization plots: Creates folder 'figures' in root directory, if it doesn't already exist.
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

for name in ["er", "ar"]:
    df = pd.read_csv(f"data/{name}_split.csv")                    # SMILES, label, and the two split columns
    embeddings = np.load(f"data/{name}_embeddings.npy")            # the 384-number row for each chemical
    labels = df["label"].values

    print(f"\n{name.upper()}")

    for split_column in ["random_split", "scaffold_split"]:
        # 1. Pick out which chemicals belong to training and to testing for this split.
        train = (df[split_column] == "train").values
        test = (df[split_column] == "test").values

        # 2. Build the classifier in two stages:
        #    - StandardScaler puts all 384 numbers on a similar scale (learned from the training chemicals only)
        #    - LogisticRegression is a simple model that learns a weight for each of the 384 numbers.
        #    class_weight="balanced" makes the rare active chemicals count for more, as in the baseline.
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

        # 3. Train it on the training chemicals only, and time it.
        start = time.time()
        classifier.fit(embeddings[train], labels[train])
        seconds = time.time() - start

        # 4. Score every chemical: a number from 0 to 1, where higher means "more likely active".
        scores = classifier.predict_proba(embeddings)[:, 1]

        # 5. Judge it on the test chemicals, which the model never saw.
        pr_auc = average_precision_score(labels[test], scores[test])
        roc_auc = roc_auc_score(labels[test], scores[test])
        no_skill = labels[test].mean()
        print(f"  {split_column:15s} PR-AUC {pr_auc:.3f} (no-skill {no_skill:.3f})   ROC-AUC {roc_auc:.3f}   trained in {seconds:.0f}s")

        # 6. Save the scores for validation and test chemicals (training rows left blank on purpose).
        df[f"frozen_{split_column.replace('_split', '')}_score"] = np.where(train, np.nan, scores)

    df.to_csv(f"data/{name}_frozen_scores.csv", index=False)

print("\nSaved data/er_frozen_scores.csv and data/ar_frozen_scores.csv")


# Visualizations: plot test-set precision–recall curves for each splitting method.

print("\nMaking visualizations ...")

# Plot test-set precision–recall curves for each splitting method.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for ax, name in zip(axes, ["er", "ar"]):
    df = pd.read_csv(f"data/{name}_frozen_scores.csv")

    # Colors and line styles distinguish methods without relying on color alone.
    for method, color, style in [
        ("random", "#0072B2", "-"),
        ("scaffold", "#D55E00", "--"),
    ]:
        test = df[df[f"{method}_split"] == "test"]

        PrecisionRecallDisplay.from_predictions(
            test["label"], test[f"frozen_{method}_score"],
            name=method.title(), color=color, linestyle=style, ax=ax,
        )

        # No-skill reference: the fraction of test chemicals that are active.
        ax.axhline(test["label"].mean(), color=color, linestyle=":",
                   label=f"{method.title()} no-skill")

    ax.set(title=name.upper(), xlabel="Recall", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(fontsize=8)

fig.suptitle("Frozen ChemBERTa: test-set precision–recall curves")
fig.tight_layout()
fig.savefig(FIGURES / "frozen_precision_recall.png", dpi=300)
plt.close(fig)

print("\nSaved figures/frozen_precision_recall.png")
