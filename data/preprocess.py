import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")

generator = rdFingerprintGenerator.GetMorganGenerator(radius = 2,fpSize = 2048)

def smiles_to_fingerprint(smiles):

    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = generator.GetFingerprint(mol)
    array = np.zeros(2048,dtype = np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array

compound_df = pd.read_csv("raw/compound.csv.gz")
compound_ids = compound_df["Metadata_JCP2022"]
smiles_list = compound_df["Metadata_SMILES"]

fingerprint_list = []
valid_compound_ids = []

for i,(compound_id,smiles) in enumerate(zip(compound_ids,smiles_list)):
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is not None:
        fingerprint_list.append(fingerprint)
        valid_compound_ids.append(compound_id)

    if (i+1) % 10000 == 0:
        print("processed: ",i+1)
fingerprints = np.stack(fingerprint_list)

print("fingerprints shape:", fingerprints.shape)
print("valid compound ids:", len(valid_compound_ids))
print("invalid smiles:", len(compound_ids) - len(valid_compound_ids))
assert len(valid_compound_ids) == fingerprints.shape[0]

valid_compound_ids = np.array(valid_compound_ids)

np.savez("processed/compound_fingerprints.npz",compound_ids=valid_compound_ids,fingerprints=fingerprints)
