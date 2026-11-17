# score_edkb.py - Step 10c: the EDKB external check. Train on Tox21, then score the EDKB chemicals.
#
# What it does: for ER and for AR, and for both the random split and the scaffold split, it
#   1. trains each model on the Tox21 TRAINING chemicals only (baseline forest, frozen ChemBERTa, fine-tuned ChemBERTa),
#   2. picks a cutoff on the Tox21 VALIDATION chemicals (the cutoff with the best F2),
#   3. scores the EDKB chemicals from prepare_edkb.py, which the models have never seen and which are not in Tox21,
#   4. reports ROC-AUC, PR-AUC, F2 and MCC on EDKB with 95% bootstrap intervals.
#
# Why this matters: Tox21 labels come from cell assays, EDKB labels come from binding assays done at the FDA. If a model
# trained on Tox21 still separates EDKB binders from non-binders, it has learned something that carries over.
#
# Read the results with care:
#   - EDKB is small after removing Tox21 overlap, so the intervals are wide.
#   - Most EDKB chemicals are binders, so a model that calls everything a binder already gets a high F2.
#     Look at MCC, ROC-AUC and PR-AUC (compared with the binder rate) first.
#
# Run it with:  python score_edkb.py
# Time:         about 40 minutes with fine-tuning (it retrains the four fine-tuned models). To skip that part and get a
#               quick answer, set RUN_FINETUNED = False below.
# It needs:     data/{er,ar}_split.csv, {er,ar}_fingerprints.npy, {er,ar}_embeddings.npy, {er,ar}_edkb.csv
# It creates:   data/edkb_metrics.csv, data/er_edkb_scores.csv, data/ar_edkb_scores.csv, and data/{er,ar}_edkb_embeddings.npy

import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

# ---- Settings you can change ----
RUN_FROZEN = True
RUN_FINETUNED = True      # the slow part (about 8 to 10 minutes per run, four runs)
SPLITS = ["random_split", "scaffold_split"]
BOOTSTRAPS = 2000
SEED = 42
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
EPOCHS, BATCH_SIZE, LEARNING_RATE = 8, 32, 3e-5   # same fine-tuning settings as train_finetune.py

rng = np.random.default_rng(SEED)


# ---------------- Metrics (same definitions as evaluate_models.py) ----------------
def counts(y, predicted):
    tp = (predicted & (y == 1)).sum(axis=-1).astype(float)
    fp = (predicted & (y == 0)).sum(axis=-1).astype(float)
    fn = (~predicted & (y == 1)).sum(axis=-1).astype(float)
    tn = (~predicted & (y == 0)).sum(axis=-1).astype(float)
    return tp, fp, fn, tn


def f2(tp, fp, fn, tn):
    denominator = 5 * tp + 4 * fn + fp
    return np.divide(5 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def mcc(tp, fp, fn, tn):
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)


def best_cutoff(y, scores):
    """The score that, used as a cutoff on the validation chemicals, gives the best F2."""
    candidates = np.unique(scores)
    return candidates[int(np.argmax([f2(*counts(y, scores >= c)) for c in candidates]))]


def evaluate(y, scores, cutoff):
    """All metrics on the EDKB chemicals, with 95% bootstrap intervals."""
    cells = counts(y, scores >= cutoff)
    idx = rng.integers(0, len(y), (BOOTSTRAPS, len(y)))
    yb, sb = y[idx], scores[idx]
    cells_b = counts(yb, sb >= cutoff)
    has_both = (yb.sum(axis=1) > 0) & (yb.sum(axis=1) < len(y))     # ROC-AUC needs both classes in the resample
    boot = {
        "F2": f2(*cells_b),
        "MCC": mcc(*cells_b),
        "PR_AUC": np.array([average_precision_score(yb[i], sb[i]) for i in range(BOOTSTRAPS)]),
        "ROC_AUC": np.array([roc_auc_score(yb[i], sb[i]) if has_both[i] else np.nan for i in range(BOOTSTRAPS)]),
    }
    observed = {"F2": float(f2(*cells)), "MCC": float(mcc(*cells)),
                "PR_AUC": average_precision_score(y, scores), "ROC_AUC": roc_auc_score(y, scores)}
    row = {}
    for m in boot:
        row[m] = observed[m]
        row[f"{m}_lo"], row[f"{m}_hi"] = np.nanpercentile(boot[m], [2.5, 97.5])
    return row


# ---------------- Turning chemicals into numbers ----------------
fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles_list):
    """Same Morgan fingerprint as build_fingerprints.py."""
    return np.array([fingerprint_generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles_list], dtype=np.uint8)


def embed(smiles_list):
    """Same frozen ChemBERTa embedding as build_embeddings.py (384 numbers per chemical)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    out = []
    for start in range(0, len(smiles_list), 64):
        tokens = tokenizer(smiles_list[start:start + 64], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            vectors = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        out.append(((vectors * mask).sum(dim=1) / mask.sum(dim=1)).numpy())
    return np.vstack(out)


def fine_tune(smiles, labels, train, val, new_smiles):
    """Same fine-tuning as train_finetune.py. Returns the best epoch's scores for the validation chemicals
    and for the new (EDKB) chemicals."""
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.chemberta = AutoModel.from_pretrained(MODEL_NAME)
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(384, 1)

        def forward(self, tokens):
            vectors = self.chemberta(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            averaged = (vectors * mask).sum(dim=1) / mask.sum(dim=1)
            return self.output(self.dropout(averaged)).squeeze(-1)

    def tokenize(s):
        return tokenizer(list(s), return_tensors="pt", padding=True, truncation=True)

    def score(model, s):
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(tokenize(s[i:i + 64]))).numpy() for i in range(0, len(s), 64)])

    torch.manual_seed(SEED)
    shuffler = np.random.default_rng(SEED)
    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = int(np.ceil(len(train) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                                    total_steps=EPOCHS * steps_per_epoch, pct_start=0.1)
    n_inactive, n_active = (labels[train] == 0).sum(), (labels[train] == 1).sum()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_inactive / n_active, dtype=torch.float32))

    best_val_pr_auc, best = -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = shuffler.permutation(train)
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            loss = loss_function(model(tokenize(smiles[batch])), torch.tensor(labels[batch], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_scores = score(model, smiles[val])
        val_pr_auc = average_precision_score(labels[val], val_scores)
        if val_pr_auc > best_val_pr_auc:                       # best epoch is chosen on validation chemicals only
            best_val_pr_auc = val_pr_auc
            best = (val_scores, score(model, new_smiles), epoch)
        print(f"      epoch {epoch}/{EPOCHS}   validation PR-AUC {val_pr_auc:.3f}")
    print(f"      kept epoch {best[2]}")
    return best[0], best[1]


# ---------------- Main loop ----------------
all_rows = []
for r in ["er", "ar"]:
    split = pd.read_csv(f"data/{r}_split.csv")
    edkb = pd.read_csv(f"data/{r}_edkb.csv")
    smiles, labels = split["smiles"].values, split["label"].values
    y = edkb["label"].values
    print(f"\n{'=' * 60}\n{r.upper()}: {len(edkb)} EDKB chemicals ({y.sum()} binders, {(y == 0).sum()} non-binders)")

    train_fp, edkb_fp = np.load(f"data/{r}_fingerprints.npy"), fingerprints(edkb["smiles"])
    if RUN_FROZEN:
        cache = f"data/{r}_edkb_embeddings.npy"
        if not os.path.exists(cache):
            np.save(cache, embed(edkb["smiles"].tolist()))
        train_emb, edkb_emb = np.load(f"data/{r}_embeddings.npy"), np.load(cache)

    scores_out = edkb[["name", "cas", "smiles", "label"]].copy()
    for sp in SPLITS:
        train = np.where(split[sp] == "train")[0]
        val = np.where(split[sp] == "val")[0]
        tag = sp.replace("_split", "")
        print(f"\n  {sp}: training on {len(train)} Tox21 chemicals")
        results = {}   # model name -> (validation scores, EDKB scores)

        start = time.time()
        forest = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=SEED)
        forest.fit(train_fp[train], labels[train])
        results["Baseline (RF)"] = (forest.predict_proba(train_fp[val])[:, 1], forest.predict_proba(edkb_fp)[:, 1])
        print(f"    baseline done ({time.time() - start:.0f}s)")

        if RUN_FROZEN:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            classifier.fit(train_emb[train], labels[train])
            results["Frozen ChemBERTa"] = (classifier.predict_proba(train_emb[val])[:, 1], classifier.predict_proba(edkb_emb)[:, 1])
            print("    frozen done")

        if RUN_FINETUNED:
            print("    fine-tuning (this is the slow part) ...")
            results["Fine-tuned ChemBERTa"] = fine_tune(smiles, labels, train, val, edkb["smiles"].values)
            results["Average (RF + fine-tuned)"] = tuple((a + b) / 2 for a, b in zip(results["Baseline (RF)"], results["Fine-tuned ChemBERTa"]))

        for model_name, (val_scores, edkb_scores) in results.items():
            cutoff = best_cutoff(labels[val], val_scores)          # cutoff chosen on Tox21 validation chemicals only
            row = dict(receptor=r.upper(), trained_on=tag, model=model_name, n_edkb=len(y), n_binders=int(y.sum()),
                       binder_rate=y.mean(), flag_everything_F2=float(f2(*counts(y, np.ones(len(y), dtype=bool)))),
                       cutoff=cutoff)
            row.update(evaluate(y, edkb_scores, cutoff))
            all_rows.append(row)
            scores_out[f"{model_name}__{tag}"] = edkb_scores
    scores_out.to_csv(f"data/{r}_edkb_scores.csv", index=False)

metrics = pd.DataFrame(all_rows)
metrics.to_csv("data/edkb_metrics.csv", index=False)

print("\nEDKB RESULTS (95% bootstrap interval in brackets)")
for _, x in metrics.iterrows():
    print(f"  {x.receptor} trained on {x.trained_on:8s} {x.model:26s} "
          f"ROC-AUC {x.ROC_AUC:.3f} [{x.ROC_AUC_lo:.3f}, {x.ROC_AUC_hi:.3f}]   "
          f"PR-AUC {x.PR_AUC:.3f} [{x.PR_AUC_lo:.3f}, {x.PR_AUC_hi:.3f}] (binder rate {x.binder_rate:.2f})   "
          f"MCC {x.MCC:.3f} [{x.MCC_lo:.3f}, {x.MCC_hi:.3f}]   F2 {x.F2:.3f} (call-everything F2 {x.flag_everything_F2:.3f})")
print("\nSaved data/edkb_metrics.csv, data/er_edkb_scores.csv and data/ar_edkb_scores.csv")
