# prepare_edkb.py - Step 10b: turn the two EDKB files into clean "binder / non-binder" tables for ER and AR.
#
# What it does: for ER and for AR, it
#   1. reads the EDKB chemicals and their binding strength (log RBA, relative binding affinity),
#   2. labels each chemical: 1 = binder, 0 = non-binder (rule explained below),
#   3. removes every chemical that is also in the Tox21 data the models trained on, so the check is truly independent,
#   4. removes repeated chemicals inside EDKB (and drops any repeat whose labels disagree),
#   5. saves the result and prints how many chemicals are left at every stage.
#
# How binders are labeled: the EDKB files carry a number (log RBA) but no binder / non-binder column. The FDA
# describes the ER set as 131 binders + 101 non-binders and the AR set as 146 binders + 56 non-binders. A non-binder
# is a chemical with no measurable binding, so it has the LOWEST log RBA values. So the lowest 101 (ER) or 56 (AR)
# chemicals are labeled non-binders and the rest binders. The script only accepts this if the cut falls cleanly
# between two different values; if chemicals are tied at the cut it stops and tells you.
#
# Run it with:  python prepare_edkb.py
# It needs:     data/edkb_er.sdf, data/edkb_ar.sdf, data/er_clean.csv, data/ar_clean.csv
# It creates:   data/er_edkb.csv, data/ar_edkb.csv, and data/er_edkb_in_tox21.csv, data/ar_edkb_in_tox21.csv
#               (the last two list the chemicals that were removed for being in Tox21)

import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # keep RDKit's warnings out of the output

SETS = {
    "ER": dict(file="data/edkb_er.sdf", rba="NCTRlogRBA", binders=131, non_binders=101, tox21="data/er_clean.csv"),
    "AR": dict(file="data/edkb_ar.sdf", rba="arNCTRlogRBA", binders=146, non_binders=56, tox21="data/ar_clean.csv"),
}
chooser = rdMolStandardize.LargestFragmentChooser()  # keeps the main molecule and drops salts and solvents


def chemical_key(molecule):
    """A short ID for 'the same chemical': the main molecule's InChIKey, ignoring 3D shape (stereochemistry).
    Two SMILES written differently for the same chemical get the same key."""
    parent = chooser.choose(molecule)
    inchi_key = Chem.MolToInchiKey(parent)
    if inchi_key:
        return inchi_key.split("-")[0]
    return Chem.MolToSmiles(parent, isomericSmiles=False)   # backup if InChI cannot be made


for name, setting in SETS.items():
    print(f"\n{'=' * 60}\n{name}")

    # 1. Read the EDKB chemicals.
    rows = []
    for molecule in Chem.SDMolSupplier(setting["file"]):
        if molecule is None:
            continue
        rows.append(dict(
            name=molecule.GetProp("Name") if molecule.HasProp("Name") else "",
            cas=molecule.GetProp("CAS") if molecule.HasProp("CAS") else "",
            smiles=Chem.MolToSmiles(molecule),
            log_rba=float(molecule.GetProp(setting["rba"])),
            key=chemical_key(molecule),
        ))
    edkb = pd.DataFrame(rows)
    expected = setting["binders"] + setting["non_binders"]
    print(f"  chemicals read from EDKB:           {len(edkb)}  (FDA says {expected})")
    if len(edkb) != expected:
        sys.exit("  The count does not match the FDA description, so I am stopping. Send me this output.")

    # 2. Label: the lowest `non_binders` values are the non-binders. Check the cut falls between two different values.
    order = np.sort(edkb["log_rba"].values)
    low, high = order[setting["non_binders"] - 1], order[setting["non_binders"]]
    print(f"  lowest log RBA values:              {np.round(order[:5], 2)}")
    print(f"  values on either side of the cut:   {low:.3f} (last non-binder)  |  {high:.3f} (first binder)")
    if low == high:
        sys.exit("  Chemicals are tied at the cut, so the label is ambiguous. Send me this output.")
    cut = (low + high) / 2
    edkb["label"] = (edkb["log_rba"] > cut).astype(int)
    print(f"  labeled:                            {edkb['label'].sum()} binders, {(edkb['label'] == 0).sum()} non-binders")

    # 3. Remove chemicals that are also in the Tox21 data this receptor's models trained on.
    tox21 = pd.read_csv(setting["tox21"])
    tox21_keys = {chemical_key(m) for m in (Chem.MolFromSmiles(s) for s in tox21["smiles"]) if m is not None}
    in_tox21 = edkb["key"].isin(tox21_keys)
    edkb[in_tox21].drop(columns="key").to_csv(f"data/{name.lower()}_edkb_in_tox21.csv", index=False)
    edkb = edkb[~in_tox21]
    print(f"  removed because they are in Tox21:  {in_tox21.sum()}")
    print(f"  left after removing Tox21 overlap:  {len(edkb)}")

    # 4. Remove repeats inside EDKB. If a repeated chemical has conflicting labels, drop it entirely.
    labels_per_chemical = edkb.groupby("key")["label"].nunique()
    conflicting = labels_per_chemical[labels_per_chemical > 1].index
    edkb = edkb[~edkb["key"].isin(conflicting)]
    before = len(edkb)
    edkb = edkb.drop_duplicates(subset="key", keep="first")
    print(f"  removed as repeats inside EDKB:     {before - len(edkb)}  (plus {len(conflicting)} chemicals with conflicting labels)")

    # 5. Save and report.
    edkb.drop(columns="key").to_csv(f"data/{name.lower()}_edkb.csv", index=False)
    print(f"  FINAL: {len(edkb)} chemicals = {edkb['label'].sum()} binders + {(edkb['label'] == 0).sum()} non-binders "
          f"({edkb['label'].mean():.1%} binders)")
    print(f"  saved data/{name.lower()}_edkb.csv")
