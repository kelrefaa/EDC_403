"""Step 4: turn every SMILES string into a 2,048-bit Morgan fingerprint.

Reads:  data/er_split.csv, data/ar_split.csv
Writes: data/er_fingerprints.npy, data/ar_fingerprints.npy
        (one row per chemical, same order as the split file)
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

SMILES_COL = "smiles"
RADIUS = 2              # radius 2 = ECFP4, the standard choice
N_BITS = 2048
DATA_DIR = "data"       # same folder clean_data.py and split_data.py use

generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

for receptor in ["er", "ar"]:
    df = pd.read_csv(f"{DATA_DIR}/{receptor}_split.csv")
    fps = np.zeros((len(df), N_BITS), dtype=np.uint8)

    for i, smiles in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Step 1 already dropped unreadable SMILES, so this should never happen
            raise ValueError(f"Row {i} could not be parsed: {smiles}")
        fps[i] = generator.GetFingerprintAsNumPy(mol)

    np.save(f"{DATA_DIR}/{receptor}_fingerprints.npy", fps)
    print(f"{receptor.upper()}: {fps.shape[0]} chemicals x {fps.shape[1]} bits, "
          f"avg {fps.sum(axis=1).mean():.1f} bits on per chemical")
