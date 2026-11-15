# split_data.py - Step 3: divide each cleaned file into train / validation / test, two different ways.
#
# What it does: for ER and for AR, it adds two columns to the cleaned data:
#   random_split   - chemicals shuffled and divided at random
#   scaffold_split - chemicals grouped into structural families, so test chemicals are structurally new
# Each column holds "train", "val" or "test" (80% / 10% / 10%).
#
# Run it with:  python split_data.py
# It creates:  data/er_split.csv  and  data/ar_split.csv

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")  # keep RDKit's warnings out of the output
SEED = 42  # fixed number so everyone on the team gets the same random split


def random_split(df):
    """Shuffle the chemicals, then cut into 80% train, 10% val, 10% test.
    'stratify' keeps the share of active chemicals about the same in every part."""
    # First cut: 80% train, 20% held back
    train, rest = train_test_split(df, train_size=0.8, stratify=df["label"], random_state=SEED)
    # Second cut: split the held-back 20% in half -> 10% val, 10% test
    val, test = train_test_split(rest, train_size=0.5, stratify=rest["label"], random_state=SEED)

    result = pd.Series(index=df.index, dtype=object)
    result[train.index] = "train"
    result[val.index] = "val"
    result[test.index] = "test"
    return result


def scaffold_split(df):
    """Group chemicals by scaffold (their ring skeleton), then fill train, val, test group by group.
    A whole family always lands in the same part, so test chemicals have scaffolds the model never trained on."""
    # 1. Find each chemical's scaffold. Chemicals with no rings get an empty scaffold and form one group.
    scaffolds = df["smiles"].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s))

    # 2. Collect the chemicals in each scaffold group, and sort the groups from biggest to smallest.
    groups = df.groupby(scaffolds).groups.values()
    groups = sorted(groups, key=len, reverse=True)

    # 3. Fill train with the biggest groups until it holds 80% of the chemicals,
    #    then val until 90%, and put all the remaining (rarer) groups in test.
    n = len(df)
    result = pd.Series(index=df.index, dtype=object)
    filled = 0
    for group in groups:
        if filled < 0.8 * n:
            part = "train"
        elif filled < 0.9 * n:
            part = "val"
        else:
            part = "test"
        result[group] = part
        filled += len(group)
    return result


def split(name, file):
    df = pd.read_csv(file)
    df["random_split"] = random_split(df)
    df["scaffold_split"] = scaffold_split(df)

    # Report how big each part is and how many actives it holds.
    print(f"\n{name}")
    for column in ["random_split", "scaffold_split"]:
        print(f"  {column}")
        for part in ["train", "val", "test"]:
            rows = df[df[column] == part]
            print(f"    {part:5s} {len(rows):5d} chemicals, {rows['label'].sum():4d} active ({rows['label'].mean():.1%})")
    return df


split("ER", "data/er_clean.csv").to_csv("data/er_split.csv", index=False)
split("AR", "data/ar_clean.csv").to_csv("data/ar_split.csv", index=False)

print("\nSaved data/er_split.csv and data/ar_split.csv")
