# inspect_edkb.py - Step 10a: look inside the two EDKB SD files before we use them.
#
# What it does: for ER and for AR, it opens the EDKB file, counts the chemicals RDKit can read, and lists every
# data field (property) the file carries, with a few example values. We need this to know which field holds the
# "binder / non-binder" answer and which holds the binding strength (RBA).
#
# Before you run it: download the two files from the "EDKB Resources" page on fda.gov
#   (Estrogen Receptor Binding Dataset -> "Download ER Binding Dataset SD File",
#    Androgen Receptor Binding Dataset -> "Download AR Binding Dataset SD File")
#   and save them as  data/edkb_er.sdf  and  data/edkb_ar.sdf
#
# Run it with:  python inspect_edkb.py
# It creates nothing. Copy what it prints and send it to me.

import os
from collections import Counter

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # keep RDKit's warnings out of the output

FILES = {"ER": "data/edkb_er.sdf", "AR": "data/edkb_ar.sdf"}

for name, path in FILES.items():
    print(f"\n{'=' * 60}\n{name}: {path}")
    if not os.path.exists(path):
        print("  file not found - check the file name and that it is inside the data folder")
        continue

    records = list(Chem.SDMolSupplier(path))          # None means RDKit could not read that record
    molecules = [m for m in records if m is not None]
    print(f"  records in the file:          {len(records)}")
    print(f"  chemicals RDKit can read:     {len(molecules)}")

    # Collect every data field and the values it takes.
    values = {}
    for m in molecules:
        for field in m.GetPropNames():
            values.setdefault(field, []).append(m.GetProp(field))

    print(f"\n  data fields ({len(values)}):")
    for field, vals in values.items():
        counts = Counter(vals)
        if len(counts) <= 6:   # few different values -> probably a class label, so show all of them
            shown = ", ".join(f"{v!r} x{n}" for v, n in counts.most_common())
        else:                  # many different values -> probably a number or a name, so show examples
            shown = "e.g. " + ", ".join(repr(v) for v in vals[:4])
        print(f"    {field:30s} filled in {len(vals):4d} times   {shown}")

    print("\n  first 3 chemicals:")
    for m in molecules[:3]:
        print(f"    SMILES {Chem.MolToSmiles(m)}")
        for field in m.GetPropNames():
            print(f"       {field}: {m.GetProp(field)}")
