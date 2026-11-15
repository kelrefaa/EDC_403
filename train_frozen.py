# train_frozen.py - Step 7b: train a simple classifier on the frozen ChemBERTa embeddings.
#
# What it does: same plan as train_baseline.py, but the input is each chemical's 384-number ChemBERTa
# embedding instead of its Morgan fingerprint, and the classifier is logistic regression instead of a forest.
# It prints the same two scores on the test chemicals, plus training time, and saves every score.
#
# Run it with:  python train_frozen.py
# It creates:   data/er_frozen_scores.csv  and  data/ar_frozen_scores.csv

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
