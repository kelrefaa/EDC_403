"""Step 9: Create project figures from saved predictions. No models are retrained."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"axes.titleweight": "bold", "figure.dpi": 120})

# Consistent CVD-friendly colors and line styles across model figures.
MODELS = {
    "Morgan + RF": ("baseline", "rf_score_{method}", "#0072B2", "-"),
    "Morgan + LR": ("baseline", "logreg_score_{method}", "#E69F00", "--"),
    "Frozen ChemBERTa": ("frozen", "frozen_{method}_score", "#009E73", "-."),
    "Fine-tuned ChemBERTa": (
        "finetuned", "finetuned_{method}_score", "#CC79A7", ":"
    ),
}
COLORS = [settings[2] for settings in MODELS.values()]
PARTS = ["train", "val", "test"]


def save(fig, filename):
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


# Load tables once.
splits = {
    assay: pd.read_csv(DATA / f"{assay}_split.csv")
    for assay in ["er", "ar"]
}
tables = {
    (assay, source): pd.read_csv(DATA / f"{assay}_{source}_scores.csv")
    for assay in splits
    for source in ["baseline", "frozen", "finetuned"]
}

# Verify that every score file uses the same rows and splitting assignments.
for (assay, source), table in tables.items():
    columns = ["smiles", "label", "random_split", "scaffold_split"]
    if not table[columns].equals(splits[assay][columns]):
        raise ValueError(f"{assay}/{source}: rows or splits do not match.")


# 1. CLASS BALANCE
counts = pd.DataFrame({
    assay.upper(): [
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    ]
    for assay, df in splits.items()
}, index=["Inactive", "Active"]).T

ax = counts.plot.bar(
    stacked=True, color=["#D9D9D9", "#E69F00"],
    figsize=(7, 4.5), rot=0, width=0.5, edgecolor="white",
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", label_type="center", fontsize=7)

for i, assay in enumerate(counts.index):
    total = counts.loc[assay].sum()
    rate = counts.loc[assay, "Active"] / total
    ax.text(i, total + 120, f"{rate:.1%} active", ha="center", fontweight="bold")

ax.set(
    title="Active vs. Inactive Chemicals in Tox21",
    xlabel="Assay", ylabel="Number of chemicals",
    ylim=(0, counts.sum(axis=1).max() * 1.18),
)
ax.legend(frameon=False)
ax.grid(axis="x", visible=False)
sns.despine(ax=ax)
save(ax.figure, "01_class_balance.png")


# 2. SPLIT COMPOSITION
from matplotlib.patches import Patch

display_parts = ["Training", "Validation", "Testing"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, (assay, df) in zip(axes, splits.items()):
    percentages = pd.DataFrame({
        method.title(): df.groupby(f"{method}_split")["label"]
                          .mean().reindex(PARTS) * 100
        for method in ["random", "scaffold"]
    })

    percentages.plot.bar(
        ax=ax,
        color=["#E69F00", "#D9D9D9"],
        edgecolor="black",
        rot=0,
        width=0.65,
        legend=False,
    )

    # Add hatch pattern to scaffold bars
    for bar in ax.containers[1]:
        bar.set_hatch("///")

    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

    # Better axis labels
    ax.set(
        title=assay.upper(),
        xlabel="Partition",
        ylabel="Active (%)",
    )
    ax.set_xticklabels(display_parts)

    # Cleaner gridlines
    ax.grid(axis="y", color="#D9D9D9", alpha=0.6)
    ax.grid(axis="x", visible=False)

    # Give room above bars
    ax.margins(y=0.25)

    sns.despine(ax=ax)

# Shared legend
legend_elements = [
    Patch(facecolor="#E69F00", edgecolor="black", label="Random"),
    Patch(facecolor="#D9D9D9", edgecolor="black", hatch="///", label="Scaffold"),
]

fig.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=False,
)

fig.suptitle(
    "Active Chemical Distribution Across Data Splits",
    weight="bold",
    y=0.98
)

fig.tight_layout(rect=[0, 0, 1, 0.90])
save(fig, "02_split_composition.png")


# Calculate AP, validation-selected thresholds, F2, and MCC.
# Store test predictions for the confusion matrices.
records = []
decisions = {}

for assay in splits:
    for method in ["random", "scaffold"]:
        for model, (source, pattern, color, style) in MODELS.items():
            df = tables[assay, source]
            column = pattern.format(method=method)
            val = df[df[f"{method}_split"] == "val"]
            test = df[df[f"{method}_split"] == "test"]

            if not np.isfinite(df.loc[val.index.union(test.index), column]).all():
                raise ValueError(f"Missing/invalid scores: {assay}, {method}, {model}")

            # Choose the threshold on VALIDATION only.
            # Candidate thresholds are the actual distinct validation scores.
            precision, recall, thresholds = precision_recall_curve(
                val["label"], val[column]
            )
            denominator = 4 * precision[:-1] + recall[:-1]
            f2 = np.divide(
                5 * precision[:-1] * recall[:-1], denominator,
                out=np.zeros_like(denominator), where=denominator > 0,
            )
            threshold = thresholds[np.argmax(f2)]

            y = test["label"].to_numpy()
            scores = test[column].to_numpy()
            predicted = scores >= threshold
            decisions[assay, method, model] = (y, predicted)

            records.append({
                "assay": assay.upper(),
                "split": method,
                "model": model,
                "AP": average_precision_score(y, scores),
                "F2": fbeta_score(y, predicted, beta=2, zero_division=0),
                "MCC": matthews_corrcoef(y, predicted),
                "threshold": threshold,
                "no_skill": y.mean(),
                "test_count": len(y),
                "test_actives": int(y.sum()),
            })

metrics = pd.DataFrame(records)
metrics.to_csv(DATA / "figure_metrics.csv", index=False)

# 3. MODEL-PERFORMANCE COMPARISON
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for row, assay in enumerate(["ER", "AR"]):
    for col, method in enumerate(["random", "scaffold"]):
        ax = axes[row, col]
        subset = metrics[
            (metrics["assay"] == assay) & (metrics["split"] == method)
        ]
        bars = ax.barh(subset["model"], subset["AP"], color=COLORS)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.axvline(
            subset["no_skill"].iloc[0], color="gray", linestyle="--",
            label=f"No-skill = {subset['no_skill'].iloc[0]:.3f}",
        )
        ax.set(title=f"{assay} · {method.title()}", xlabel="Average Precision (PR-AUC)")
        ax.set_xlim(0, metrics["AP"].max() * 1.3)
        ax.invert_yaxis()
        ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=ax)

fig.suptitle("Model Performance Comparison: PR-AUC", weight="bold")
save(fig, "03_model_average_precision.png")


# 4. SCAFFOLD F2 AND MCC

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for row, assay in enumerate(["ER", "AR"]):
    subset = metrics[
        (metrics["assay"] == assay) &
        (metrics["split"] == "scaffold")
    ].set_index("model").reindex(MODELS)

    for col, metric in enumerate(["F2", "MCC"]):
        ax = axes[row, col]
        values = subset[metric]

        # Set axis limits appropriate for each panel
        if metric == "F2":
            xlim = (0, 0.65) if assay == "ER" else (0, 0.10)
        else:
            xlim = (-0.10, 0.35) if assay == "ER" else (-0.10, 0.10)

        bars = ax.barh(
            subset.index,
            values,
            color=COLORS,
            edgecolor="white",
            height=0.75
        )

        ax.set_xlim(*xlim)
        if assay == "AR" and metric == "MCC":
            ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10])

        # Add labels with space around positive and negative values
        offset = (xlim[1] - xlim[0]) * 0.015

        for bar, value in zip(bars, values):
            y = bar.get_y() + bar.get_height() / 2

            if value > 0:
                ax.text(
                    value + offset, y,
                    f"{value:.3f}",
                    va="center", ha="left",
                    fontsize=10
                )
            elif value < 0:
                ax.text(
                    value - offset, y,
                    f"{value:.3f}",
                    va="center", ha="right",
                    fontsize=10
                )
            else:
                ax.text(
                    offset, y,
                    "0.000",
                    va="center", ha="left",
                    fontsize=10
                )

        # Zero reference line
        ax.axvline(
            0, color="gray",
            linewidth=1, linestyle="-"
        )

        ax.set(
            title=f"{assay} · Scaffold",
            xlabel=metric
        )

        # Improve gridlines
        ax.grid(axis="x", color="#D9D9D9", alpha=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

        # Keep model ordering consistent
        ax.invert_yaxis()
        sns.despine(ax=ax)

        # Highlight the hypothesis-related ER F2 difference
        if assay == "ER" and metric == "F2":
            delta = (
                subset.loc["Fine-tuned ChemBERTa", "F2"]
                - subset.loc["Morgan + RF", "F2"]
            )

            ax.set_title(
                f"ER · Scaffold · ΔF2 = {delta:+.3f} "
                "(target ≥ 0.05)",
                fontsize=11,
                fontweight="bold"
            )

fig.suptitle(
    "Model Performance on Scaffold Test Sets: F2 and MCC",
    fontsize=15,
    fontweight="bold",
    y=0.99
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

save(fig, "04_scaffold_f2_mcc.png")



# 5. COMBINED PRECISION–RECALL CURVES

fig, axes = plt.subplots(
    2, 2, figsize=(13, 9),
    sharex=True, sharey=True
)

for row, assay in enumerate(["er", "ar"]):
    for col, method in enumerate(["random", "scaffold"]):
        ax = axes[row, col]

        for model, (source, pattern, color, style) in MODELS.items():
            df = tables[assay, source]
            test = df[df[f"{method}_split"] == "test"]

            y = test["label"]
            scores = test[pattern.format(method=method)]

            precision, recall, _ = precision_recall_curve(
                y, scores
            )
            ap = average_precision_score(y, scores)

            ax.step(
                recall, precision,
                where="post",
                color=color,
                linestyle="-",
                linewidth=2.2,
                label=f"{model} (AP = {ap:.3f})",
            )

        # No-skill reference line
        no_skill = y.mean()

        ax.axhline(
            no_skill,
            color="#777777",
            linestyle="--",
            linewidth=1.5,
            label=f"No-skill ({no_skill:.3f})",
        )

        # Panel titles
        ax.set_title(
            f"{assay.upper()} · {method.title()} "
            f"· {int(y.sum())} actives",
            fontsize=12,
            fontweight="bold",
            pad=10
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        # Show labels only on outer edges
        ax.set_xlabel("Recall" if row == 1 else "")
        ax.set_ylabel("Precision" if col == 0 else "")

        # Lighter gridlines
        ax.grid(
            color="#D9D9D9",
            alpha=0.5,
            linewidth=0.8
        )
        ax.set_axisbelow(True)

        # Keep model labels outside the plotting area
        ax.legend(
            fontsize=8,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
        )

        sns.despine(ax=ax)

fig.suptitle(
    "Precision–Recall Curves Across Models",
    fontsize=16,
    fontweight="bold",
    y=0.99
)

# Leave room for legends beneath each panel
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.91,
    bottom=0.12,
    hspace=0.65,
    wspace=0.18
)

save(fig, "05_combined_precision_recall.png")



# 6. ER SCAFFOLD CONFUSION MATRICES
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, model in zip(axes, ["Morgan + RF", "Fine-tuned ChemBERTa"]):
    y, predicted = decisions["er", "scaffold", model]
    matrix = confusion_matrix(y, predicted, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        matrix, display_labels=["Inactive", "Active"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        im_kw={"vmin": 0, "vmax": 384}
    )

    # Add outcome names to displayed counts
    for text, label in zip(display.text_.flat, ["TN", "FP", "FN", "TP"]):
        text.set_text(f"{label}\n{text.get_text()}")

    ax.set_title(model, fontweight="bold")
    ax.grid(False)

fig.suptitle(
    "ER Scaffold Test: Confusion Matrix Comparison",
    fontsize=14,
    fontweight="bold",
    y=1.06
)

fig.text(
    0.5, 0.99,
    "Classification thresholds selected using validation F2-score",
    ha="center",
    fontsize=10
)

fig.tight_layout()
save(fig, "06_er_scaffold_confusion.png")

print("\nAll figures saved in the 'figures' folder.")
