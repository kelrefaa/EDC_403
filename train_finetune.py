# train_finetune.py - Step 8: fine-tune ChemBERTa on our ER / AR labels.
#
# What it does: for ER and for AR, and for both the random split and the scaffold split, it retrains ALL of
# ChemBERTa's weights on the training chemicals (that is what "fine-tuning" means), keeps the best epoch
# (chosen on the validation chemicals only), then scores the test chemicals once.
# It prints the same two scores as the other models, plus training time.
#
# This is the slowest step. To check how long it takes, set EPOCHS = 1 below for a first try.
#
# Run it with:  python train_finetune.py
# It creates:   data/er_finetuned_scores.csv  and  data/ar_finetuned_scores.csv

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

# ---- Settings you can change ----
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
EPOCHS = 8            # how many times the model sees the whole training set
BATCH_SIZE = 32       # how many chemicals it learns from at once
LEARNING_RATE = 3e-5  # how big each learning step is (small, because we are adjusting a pretrained model)
SEED = 42             # fixed number so results are repeatable

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class Classifier(nn.Module):
    """ChemBERTa plus one small output layer that gives a single number per chemical."""

    def __init__(self):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(MODEL_NAME)  # the pretrained model
        self.dropout = nn.Dropout(0.1)                          # randomly ignores a few values while training, to avoid overfitting
        self.output = nn.Linear(384, 1)                         # turns the 384-number embedding into 1 number

    def forward(self, tokens):
        # Same steps as build_embeddings.py: run ChemBERTa, then average the token vectors (ignoring padding)
        vectors = self.chemberta(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        averaged = (vectors * mask).sum(dim=1) / mask.sum(dim=1)
        return self.output(self.dropout(averaged)).squeeze(-1)


def tokenize(smiles):
    """Chop SMILES strings into tokens the model can read."""
    return tokenizer(list(smiles), return_tensors="pt", padding=True, truncation=True)


def score(model, smiles):
    """Give each chemical a score from 0 to 1 (higher = more likely active)."""
    model.eval()  # evaluation mode: no learning, no dropout
    results = []
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            results.append(torch.sigmoid(model(tokenize(smiles[i:i + 64]))).numpy())
    return np.concatenate(results)


def fine_tune(df, split_column):
    """Fine-tune one model on one split. Returns a score for every validation and test chemical."""
    smiles = df["smiles"].values
    labels = df["label"].values
    train = np.where(df[split_column] == "train")[0]   # positions of the training chemicals
    val = np.where(df[split_column] == "val")[0]
    test = np.where(df[split_column] == "test")[0]

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # The learning rate starts small, rises briefly, then shrinks. This is a common way to keep fine-tuning stable.
    steps_per_epoch = int(np.ceil(len(train) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS * steps_per_epoch, pct_start=0.1
    )

    # Active chemicals are rare, so we make mistakes on them count for more (like class_weight="balanced").
    n_inactive, n_active = (labels[train] == 0).sum(), (labels[train] == 1).sum()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_inactive / n_active, dtype=torch.float32))

    best_val_score = -1
    best_scores = None
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # 1. Learn: go through the training chemicals in random order, a batch at a time.
        model.train()
        total_loss = 0
        order = rng.permutation(train)
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            predictions = model(tokenize(smiles[batch]))
            loss = loss_function(predictions, torch.tensor(labels[batch], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops any single step from being too big
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)

        # 2. Check: score the validation chemicals. If this is the best epoch so far, remember its scores.
        val_scores = score(model, smiles[val])
        val_pr_auc = average_precision_score(labels[val], val_scores)
        if val_pr_auc > best_val_score:
            best_val_score, best_epoch = val_pr_auc, epoch
            best_scores = (val_scores, score(model, smiles[test]))

        print(f"    epoch {epoch}/{EPOCHS}   training loss {total_loss / len(train):.3f}   "
              f"validation PR-AUC {val_pr_auc:.3f}   ({time.time() - start:.0f}s so far)")

    # 3. Put the best epoch's scores into one list that lines up with the table's rows.
    all_scores = np.full(len(df), np.nan)
    all_scores[val], all_scores[test] = best_scores
    return all_scores, test, best_epoch, time.time() - start


for name in ["er", "ar"]:
    df = pd.read_csv(f"data/{name}_split.csv")
    print(f"\n{name.upper()}")

    for split_column in ["random_split", "scaffold_split"]:
        print(f"  {split_column}")
        scores, test, best_epoch, seconds = fine_tune(df, split_column)
        df[f"finetuned_{split_column.replace('_split', '')}_score"] = scores

        # Judge on the test chemicals, which were scored only once, at the best epoch.
        labels = df["label"].values[test]
        pr_auc = average_precision_score(labels, scores[test])
        roc_auc = roc_auc_score(labels, scores[test])
        print(f"  -> TEST: PR-AUC {pr_auc:.3f} (no-skill {labels.mean():.3f})   ROC-AUC {roc_auc:.3f}   "
              f"best epoch {best_epoch}   trained in {seconds / 60:.1f} min")

    df.to_csv(f"data/{name}_finetuned_scores.csv", index=False)

print("\nSaved data/er_finetuned_scores.csv and data/ar_finetuned_scores.csv")
