# build_embeddings.py - Step 7a: turn every chemical into a ChemBERTa embedding (frozen, no training).
#
# What it does: for ER and for AR, it feeds each chemical's SMILES through the pretrained ChemBERTa model
# and saves one list of 384 numbers per chemical. ChemBERTa itself is NOT changed ("frozen").
#
# Run it with:  python build_embeddings.py     (takes a few minutes)
# It creates:   data/er_embeddings.npy  and  data/ar_embeddings.npy
# Row 1 of each file belongs to row 1 of er_split.csv / ar_split.csv, row 2 to row 2, and so on.

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
BATCH_SIZE = 64  # how many chemicals to process at once

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # "evaluation mode": we are only reading from the model, not training it


def embed(smiles_list):
    """Turn a list of SMILES strings into a table with 384 numbers per chemical."""
    all_embeddings = []
    for start in range(0, len(smiles_list), BATCH_SIZE):
        batch = smiles_list[start:start + BATCH_SIZE]

        # 1. Chop each SMILES into tokens. Shorter ones are padded so the batch lines up;
        #    very long ones are cut off at the model's length limit.
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        # 2. Run the model. It returns one vector of 384 numbers for every token.
        with torch.no_grad():  # no training, so no need to track changes
            vectors = model(**tokens).last_hidden_state

        # 3. Average the token vectors into one vector per chemical.
        #    The mask makes sure the padding tokens don't count toward the average.
        mask = tokens["attention_mask"].unsqueeze(-1)
        averaged = (vectors * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(averaged.numpy())

        if (start // BATCH_SIZE) % 20 == 0:
            print(f"    {start} / {len(smiles_list)}")
    return np.vstack(all_embeddings)


for name in ["er", "ar"]:
    df = pd.read_csv(f"data/{name}_split.csv")
    print(f"\n{name.upper()}: embedding {len(df)} chemicals ...")
    embeddings = embed(df["smiles"].tolist())
    np.save(f"data/{name}_embeddings.npy", embeddings)
    print(f"  table size: {embeddings.shape[0]} chemicals x {embeddings.shape[1]} numbers")

print("\nSaved data/er_embeddings.npy and data/ar_embeddings.npy")
