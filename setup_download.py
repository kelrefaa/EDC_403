# setup_download.py - one-time setup. Downloads the data and model, then tests them.
#
# Run in the terminal:
#   pip install pandas numpy scikit-learn rdkit torch transformers
#   python setup_download.py

import urllib.request
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoModel, AutoTokenizer

TOX21_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz"
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
SMILES = "COC(=O)c1ccc(O)cc1"  # methylparaben, used for the tests

# 1. Tox21 data: download once into ./data/, then print how many labels we have
Path("data").mkdir(exist_ok=True)
tox21_file = Path("data/tox21.csv.gz")
if not tox21_file.exists():
    urllib.request.urlretrieve(TOX21_URL, tox21_file)
df = pd.read_csv(tox21_file)
print(f"Tox21: {len(df)} chemicals")
for assay in ["NR-ER", "NR-AR"]:
    labels = df[assay].dropna()
    print(f"  {assay}: {len(labels)} labeled, {labels.mean():.1%} active")

# 2. RDKit test: turn one molecule into a Morgan fingerprint
mol = Chem.MolFromSmiles(SMILES)
fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol)
print(f"RDKit works: fingerprint has {len(fingerprint)} positions, {int(fingerprint.sum())} set to 1")

# 3. ChemBERTa: download (cached after the first time), then embed one molecule
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
with torch.no_grad():
    embedding = model(**tokenizer(SMILES, return_tensors="pt")).last_hidden_state.mean(dim=1)
print(f"ChemBERTa works: embedding shape {tuple(embedding.shape)} (expect 1 x 384)")
