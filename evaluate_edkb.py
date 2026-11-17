"""Evaluate saved Tox21 models on non-overlapping EDKB chemicals.
Run after training: python evaluate_edkb.py
Requires data/{er,ar}_split.csv, data/edkb_{er,ar}_labeled.csv,
models/{er,ar}_{frozen,finetuned}_{random,scaffold}.*, and ChemBERTa.
Uses the validation-set F2-optimal threshold, never EDKB labels, to select threshold.
PR-AUC is sklearn average precision, as in the original project.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef
from transformers import AutoModel, AutoTokenizer
from train_finetune_save import Classifier, score as score_finetuned

DATA = Path("data")
MODELS = Path("models")
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def canonical(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None


def get_embeddings(smiles, tokenizer, backbone):
    result = []
    backbone.eval()
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            inputs = tokenizer(list(smiles[i:i + 64]), return_tensors="pt", padding=True, truncation=True)
            vectors = backbone(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1)
            result.append(pooled.numpy())
    return np.vstack(result)


def threshold_from_val(labels, val_scores):
    values = [fbeta_score(labels, val_scores >= threshold, beta=2, zero_division=0)
              for threshold in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(values))])


def metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return dict(F2=float(fbeta_score(labels, pred, beta=2, zero_division=0)),
                MCC=float(matthews_corrcoef(labels, pred)) if len(np.unique(labels)) == 2 else np.nan,
                PR_AUC_average_precision=float(average_precision_score(labels, scores))
                    if len(np.unique(labels)) == 2 else np.nan)


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = None
    results = []
    DATA.mkdir(exist_ok=True)
    for assay in ("er", "ar"):
        tox = pd.read_csv(DATA / f"{assay}_split.csv")
        edkb = pd.read_csv(DATA / f"edkb_{assay}_labeled.csv")
        for df, label in [(tox, "Tox21"), (edkb, "EDKB")]:
            if not {"smiles", "label"}.issubset(df.columns):
                raise ValueError(f"{label} missing smiles or label")
        raw_tox_path = DATA / "tox21.csv.gz"
        if raw_tox_path.exists():
            tox_keys = set(pd.read_csv(raw_tox_path)["smiles"].map(canonical).dropna())
        else:
            print("WARNING: data/tox21.csv.gz missing; checking overlap only against cleaned Tox21 assay file")
            tox_keys = set(tox.smiles.map(canonical).dropna())
        edkb["canonical_smiles"] = edkb.smiles.map(canonical)
        invalid = edkb.canonical_smiles.isna().sum()
        overlap = edkb.canonical_smiles.isin(tox_keys)
        removed_overlap = int(overlap.sum())
        edkb = edkb.loc[edkb.canonical_smiles.notna() & ~overlap].copy()
        if edkb.canonical_smiles.duplicated().any():
            raise ValueError(f"Duplicate EDKB structures in {assay}; inspect manually")
        if len(edkb) == 0:
            raise ValueError(f"No non-overlapping EDKB chemicals for {assay}")
        print(f"{assay.upper()}: {len(edkb)} external records after {removed_overlap} Tox21 overlaps and {invalid} invalid records removed")
        edkb_scores = None
        for model_type in ("frozen", "finetuned"):
            for split in ("random", "scaffold"):
                split_col = f"{split}_split"
                valid = tox[split_col].eq("val").to_numpy()
                labels_val = tox.loc[valid, "label"].to_numpy(dtype=int)
                if model_type == "frozen":
                    saved = MODELS / f"{assay}_frozen_{split}.joblib"
                    if not saved.exists():
                        print(f"SKIPPING missing {saved}")
                        continue
                    classifier = joblib.load(saved)  # Load only your own trusted files.
                    if backbone is None:
                        backbone = AutoModel.from_pretrained(MODEL_NAME)
                    val_embed = get_embeddings(tox.loc[valid, "smiles"].tolist(), tokenizer, backbone)
                    if edkb_scores is None:
                        edkb_scores = get_embeddings(edkb.smiles.tolist(), tokenizer, backbone)
                    val_scores = classifier.predict_proba(val_embed)[:, 1]
                    ext_scores = classifier.predict_proba(edkb_scores)[:, 1]
                else:
                    saved = MODELS / f"{assay}_finetuned_{split}.pt"
                    if not saved.exists():
                        print(f"SKIPPING missing {saved}")
                        continue
                    network = Classifier()
                    network.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True))
                    val_scores = score_finetuned(network, tox.loc[valid, "smiles"].to_numpy())
                    ext_scores = score_finetuned(network, edkb.smiles.to_numpy())
                    del network
                threshold = threshold_from_val(labels_val, val_scores)
                row = {"assay": assay.upper(), "model": model_type, "split": split,
                       "n_external": len(edkb), "n_removed_tox21_overlap": removed_overlap,
                       "threshold_selected_on_tox21_val": threshold,
                       **metrics(edkb.label.to_numpy(dtype=int), ext_scores, threshold)}
                results.append(row)
                out = edkb.drop(columns=["canonical_smiles"]).copy()
                out["prediction_score"] = ext_scores
                out["predicted_label"] = (ext_scores >= threshold).astype(int)
                out.to_csv(DATA / f"edkb_{assay}_{model_type}_{split}_predictions.csv", index=False)
                print(row)
    if not results:
        raise SystemExit("No model checkpoints found. Run train_frozen_save.py / train_finetune_save.py first.")
    pd.DataFrame(results).to_csv(DATA / "edkb_external_metrics.csv", index=False)
    print("Saved data/edkb_external_metrics.csv and individual prediction CSV files")


if __name__ == "__main__":
    run()
