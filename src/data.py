import numpy as np
import torch

from torch.utils.data import TensorDataset,DataLoader

def load_data(batch_size=32):
    data = np.load("data/processed/paired_data.npz")

    compound_ids = data["compound_ids"]
    fingerprints = data["fingerprints"]
    phenotypes = data["phenotypes"]

    np.random.seed(31)

    indices = np.arange(len(compound_ids))
    np.random.shuffle(indices)

    train_end = int(len(indices) * 0.8)
    val_end = int(len(indices) * 0.9)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]


    X_mol_train = fingerprints[train_indices]
    X_pheno_train = phenotypes[train_indices]

    X_mol_val = fingerprints[val_indices]
    X_pheno_val = phenotypes[val_indices]

    X_mol_test = fingerprints[test_indices]
    X_pheno_test = phenotypes[test_indices]


    X_mol_train = torch.tensor(X_mol_train, dtype=torch.float32)
    X_pheno_train = torch.tensor(X_pheno_train, dtype=torch.float32)

    X_mol_val = torch.tensor(X_mol_val, dtype=torch.float32)
    X_pheno_val = torch.tensor(X_pheno_val, dtype=torch.float32)

    X_mol_test = torch.tensor(X_mol_test, dtype=torch.float32)
    X_pheno_test = torch.tensor(X_pheno_test, dtype=torch.float32)


    train_dataset = TensorDataset(X_mol_train,X_pheno_train)

    val_dataset = TensorDataset(X_mol_val,X_pheno_val)

    test_dataset = TensorDataset(X_mol_test,X_pheno_test)


    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader


    # merged_df = pd.merge(molecule_df,phenotype_df,how="inner",on="compound_id")
    #
    # print("merged data shape: ",merged_df.shape)
    #
    # merged_df = merged_df.dropna()
    #
    # molecule_columns = []
    # for column in merged_df.columns:
    #     if column.startswith("fp_"):
    #         molecule_columns.append(column)
    #
    # phenotype_columns = []
    # for column in merged_df.columns:
    #     if column.startswith("p_"):
    #         phenotype_columns.append(column)
    #
    # print("molecule feature numbers: ",len(molecule_columns))
    # print("phenotype features numbers: ",len(phenotype_columns))
    #
    # X_mol = torch.tensor(merged_df[molecule_columns].to_numpy(),dtype=torch.float32)
    # X_ph = torch.tensor(merged_df[phenotype_columns].to_numpy(),dtype=torch.float32)
    #
    #
    # assert X_mol.shape[0] == X_ph.shape[0]
    #
    # assert not torch.isnan(X_mol).any()
    # assert not torch.isnan(X_ph).any()
    #
    #
    # dataset = TensorDataset(X_mol,X_ph)
    #
    # dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    #
    # return dataloader
