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

# For visualization plots: Creates folder 'figures' in root directory, if it doesn't already exist.
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

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

# Visualizations:
print("\nMaking visualizations ...")

# Chart 1: compare active and inactive counts in the cleaned datasets.

# Count labels for each assay: 0 = inactive, 1 = active.
counts = pd.DataFrame(
    {
        "Inactive": [
            (er_clean["label"] == 0).sum(),
            (ar_clean["label"] == 0).sum(),
        ],
        "Active": [
            (er_clean["label"] == 1).sum(),
            (ar_clean["label"] == 1).sum(),
        ],
    },
    index=["NR-ER", "NR-AR"],
)

# Use CVD-friendly colors and patterns to distinguish the categories.
ax = counts.plot.bar(
    stacked=True,
    color=["#D9D9D9", "#E69F00"],
    edgecolor="black",
    figsize=(7, 5),
    rot=0,
)


# Label the chart and axes.
ax.set_title("Active and inactive chemicals after cleaning")
ax.set_xlabel("Assay")
ax.set_ylabel("Number of chemicals")
ax.legend(title="Label")

# Label each section with its chemical count.
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", label_type="center")

# Above each bar, show the percentage of labeled chemicals that are active.
for position, assay in enumerate(counts.index):
    total = counts.loc[assay].sum()
    active_percent = counts.loc[assay, "Active"] / total * 100

    ax.text(
        position,
        total + 100,
        f"{active_percent:.1f}% active",
        ha="center",
    )

# Leave room above the bars for the percentage labels.
ax.set_ylim(0, counts.sum(axis=1).max() * 1.15)

# Fit the labels within the image, save it, then close the figure.
plt.tight_layout()
plt.savefig(FIGURES / "assay_activity.png", dpi=300)
plt.close()

print("\nSaved figures/assay_activity.png")

# Chart 2: available vs. missing labels in the raw dataset.

availability = pd.DataFrame({
    "Available": raw[["NR-ER", "NR-AR"]].notna().sum(),
    "Missing": raw[["NR-ER", "NR-AR"]].isna().sum(),
})

ax = availability.plot.bar(stacked=True, color=["#E69F00", "#D9D9D9"],
                           edgecolor="black", rot=0, figsize=(7, 5))
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", label_type="center")
ax.set(title="Label availability before cleaning", xlabel="Assay",
       ylabel="Number of chemicals")
plt.tight_layout()
plt.savefig(FIGURES / "label_availability.png", dpi=300)
plt.close()

print("Saved figures/label_availability.png")

