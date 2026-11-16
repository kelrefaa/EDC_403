# split_data.py - Step 3: divide each cleaned file into train / validation / test, two different ways.
#
# What it does: for ER and for AR, it adds two columns to the cleaned data:
#   random_split   - chemicals shuffled and divided at random
#   scaffold_split - chemicals grouped into structural families, so test chemicals are structurally new
# Each column holds "train", "val" or "test" (80% / 10% / 10%).
#
# Run it with:  python split_data.py
# It creates:  data/er_split.csv  and  data/ar_split.csv

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# For visualization plots: Creates folder 'figures' in root directory, if it doesn't already exist.
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

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


# Visualizations:

print("\nMaking visualizations ...")

# Load the split datasets.
datasets = {name: pd.read_csv(f"data/{name.lower()}_split.csv")
            for name in ["ER", "AR"]}
parts = ["train", "val", "test"]

# Make one figure for counts and another for active percentages.
for metric in ["size", "mean"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (name, df) in zip(axes, datasets.items()):
        values = pd.DataFrame({
            method: df.groupby(f"{method.lower()}_split")["label"]
                      .agg(metric).reindex(parts)
            for method in ["Random", "Scaffold"]
        })
        if metric == "mean":
            values *= 100

        values.plot.bar(ax=ax, color=["#E69F00", "#D9D9D9"],
                        edgecolor="black", rot=0)
        for bar in ax.containers[1]:
            bar.set_hatch("///")
        for container in ax.containers:
            ax.bar_label(container,
                         fmt="%.0f" if metric == "size" else "%.1f%%",
                         fontsize=8)

        ax.set(title=name, xlabel="Partition")
        ax.set_ylabel("Number of chemicals" if metric == "size" else "Active (%)")
        ax.margins(y=0.2)
        ax.legend()

    fig.suptitle("Split sizes" if metric == "size" else "Active percentage by split")
    fig.tight_layout()
    fig.savefig(FIGURES / f"split_{metric}.png", dpi=300)
    plt.close(fig)


print("\nSaved figures/split_size.png and figures/split_mean.png")