import numpy as np
import pandas as pd


compound_data = np.load("processed/compound_fingerprints.npz")

compound_ids = compound_data["compound_ids"]
fingerprints = compound_data["fingerprints"]

phenotype_df = pd.read_parquet("raw/compound_profiles.parquet")

phenotype_columns = []

for column in phenotype_df.columns:
    if column.startswith("X_"):
        phenotype_columns.append(column)

phenotype_df = phenotype_df[phenotype_df["Metadata_JCP2022"] != "JCP2022_033924"]
# 033924为DMSO阴性对照

phenotype_mean = (phenotype_df.groupby("Metadata_JCP2022")[phenotype_columns].mean())

mask = np.isin(compound_ids,phenotype_mean.index)

paired_ids = compound_ids[mask]
paired_fingerprints = fingerprints[mask]

paired_phenotypes = phenotype_mean.loc[paired_ids,phenotype_columns].to_numpy(dtype=np.float32)

print("paired compound number:", len(paired_ids))
print("paired fingerprints shape:", paired_fingerprints.shape)
print("paired phenotypes shape:", paired_phenotypes.shape)

assert len(paired_ids) == paired_fingerprints.shape[0]
assert len(paired_ids) == paired_phenotypes.shape[0]

assert not np.isnan(paired_fingerprints).any()
assert not np.isnan(paired_phenotypes).any()

np.savez("processed/paired_data.npz",compound_ids=paired_ids,fingerprints=paired_fingerprints,phenotypes=paired_phenotypes)
