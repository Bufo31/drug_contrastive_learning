# Drug Contrastive Learning

A small PyTorch project for learning a shared embedding space between molecular structures and cell-phenotype profiles with a dual-encoder contrastive-learning model.

This repository currently records the first working **baseline** on real paired data. Later evaluation and retrieval improvements will be added as separate commits so that the development process remains visible in Git history.

## Pipeline

1. Read compound IDs and SMILES.
2. Convert valid SMILES to 2048-bit Morgan fingerprints (`radius=2`).
3. Aggregate phenotype profiles by `Metadata_JCP2022`.
4. Remove the DMSO negative-control ID `JCP2022_033924`.
5. Match molecular fingerprints with phenotype vectors.
6. Split paired samples into train / validation / test sets (80% / 10% / 10%).
7. Train two MLP encoders with a symmetric contrastive loss.

## Project structure

```text
drug_contrastive_learning/
├── data/
│   ├── preprocess.py          # SMILES -> Morgan fingerprints
│   ├── pairs.py               # build paired molecular/phenotype data
│   ├── raw/                   # raw datasets (not tracked by Git)
│   └── processed/             # generated .npz files (not tracked by Git)
├── src/
│   ├── data.py                # split data and build DataLoaders
│   ├── model.py               # MLP encoders and DualEncoder
│   ├── loss.py                # symmetric contrastive loss
│   ├── train.py               # training loop
│   └── main.py                # baseline training entry point
├── test.py                    # parquet structure/statistics inspection script
├── parquet_report.txt         # dataset structure report from the inspection step
└── .gitignore
```

## Data

Large raw and processed data files are intentionally excluded from GitHub.

Place the raw files under:

```text
data/raw/compound.csv.gz
data/raw/compound_profiles.parquet
```

The preprocessing scripts generate:

```text
data/processed/compound_fingerprints.npz
data/processed/paired_data.npz
```

The phenotype table used in this baseline contains 737 numeric phenotype features. Molecular inputs are 2048-dimensional Morgan fingerprints.

## Running the baseline

The current preprocessing scripts use paths relative to the `data/` directory:

```bash
cd data
python preprocess.py
python pairs.py
cd ..
```

Then start training from the project root:

```bash
python src/main.py
```

## Baseline configuration

- Molecular input dimension: 2048
- Phenotype input dimension: 737
- Hidden dimension: 512
- Embedding dimension: 128
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 5
- Device used for the first run: CPU

First successful real-data training run:

```text
Epoch 1: 2.9540
Epoch 2: 2.4127
Epoch 3: 1.9290
Epoch 4: 1.4891
Epoch 5: 1.1487
```

These values are training losses only. Validation, test-set evaluation, and retrieval metrics are planned as later stages rather than being reported as baseline performance.

## Dependencies

Main Python packages used by the current code:

```text
numpy
pandas
pyarrow
rdkit
torch
```

## Current status

- [x] Real dataset preprocessing
- [x] SMILES to Morgan fingerprints
- [x] Molecular / phenotype pairing
- [x] Train / validation / test split
- [x] First real-data dual-encoder training run
- [ ] Validation and test evaluation
- [ ] Retrieval metrics
- [ ] Training-speed and hyperparameter optimization
