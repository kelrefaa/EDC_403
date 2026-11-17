# clean_data.py - Step 2: turn the raw Tox21 table into a clean list of (chemical, label) pairs.
#
# What it does: for ER and for AR, it keeps that receptor's labels, drops chemicals with no label or
# an unreadable SMILES, and saves the result as er_clean.csv and ar_clean.csv.
#
# Run it with:  python clean_data.py
# It creates two files in ./data/:  er_clean.csv  and  ar_clean.csv

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger


# RDKit prints a lot of warnings for unusual molecules. We count the unreadable ones ourselves, so silence them.
RDLogger.DisableLog("rdApp.*")

# Load the raw table that setup_download.py saved.
# Each row is one chemical: a "smiles" column plus 12 assay columns (1 = active, 0 = inactive, blank = not tested).
raw = pd.read_csv("data/tox21.csv.gz")


def clean(assay):
    """Clean the table for ONE assay (e.g. "NR-ER") and return a table with two columns: smiles, label."""

    # 1. Keep only the chemical's structure (smiles) and the one assay we care about.
    df = raw[["smiles", assay]].rename(columns={assay: "label"})
    print(f"\n{assay}")
    print(f"  chemicals in the raw table:   {len(df)}")

    # 2. Drop chemicals that were never tested in this assay (blank label).
    #    With no answer, there is nothing for the model to learn from.
    df = df.dropna(subset=["label"])
    print(f"  after dropping blank labels:  {len(df)}")

    # 3. Check that RDKit can read each SMILES as a real molecule.
    #    If a SMILES can't be read (MolFromSmiles returns None), we drop that chemical.
    can_read = [Chem.MolFromSmiles(s) is not None for s in df["smiles"]]
    df = df[can_read]
    print(f"  after dropping unreadable:    {len(df)}")

    # 4. Make the label a whole number (1 or 0) instead of 1.0 or 0.0.
    df["label"] = df["label"].astype(int)

    # 5. Report how many are active. This is the "active rate".
    print(f"  active: {df['label'].sum()} ({df['label'].mean():.1%})")
    return df


# Clean each receptor's data and keep the resulting DataFrames
# Save them in memory so we can also use them for visualizations.
er_clean = clean("NR-ER")
ar_clean = clean("NR-AR")

# Save each DataFrame as a CSV file.
# index=False leaves out pandas' extra row-number column.
er_clean.to_csv("data/er_clean.csv", index=False)
ar_clean.to_csv("data/ar_clean.csv", index=False)

print("\nSaved data/er_clean.csv and data/ar_clean.csv")