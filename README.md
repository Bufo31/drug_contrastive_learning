# Drug Contrastive Learning

A small PyTorch project for learning a shared embedding space between molecular structures and cell-phenotype profiles with a dual-encoder contrastive-learning model.

This repository records the project step by step so that the baseline, evaluation pipeline, and later optimization process remain visible in Git history.

## Pipeline

1. Read compound IDs and SMILES.
2. Convert valid SMILES to 2048-bit Morgan fingerprints (`radius=2`).
3. Aggregate phenotype profiles by `Metadata_JCP2022`.
4. Remove the DMSO negative-control ID `JCP2022_033924`.
5. Match molecular fingerprints with phenotype vectors.
6. Split paired samples into train / validation / test sets (80% / 10% / 10%).
7. Train two MLP encoders with a symmetric contrastive loss.
8. Select the best checkpoint using validation loss.
9. Evaluate cross-modal retrieval in both directions with Recall@1 / Recall@5 / Recall@10.

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
│   ├── train.py               # training + validation + best checkpoint
│   ├── evaluate.py            # bidirectional retrieval evaluation
│   └── main.py                # switch between train and evaluate modes
├── test.py                    # parquet structure/statistics inspection script
├── parquet_report.txt         # dataset structure report from the inspection step
└── .gitignore
```

## Data

Large raw datasets, generated `.npz` files, and model checkpoints are intentionally excluded from GitHub.

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

The phenotype table contains 737 numeric phenotype features. Molecular inputs are 2048-dimensional Morgan fingerprints.

## Running

Run preprocessing from the `data/` directory:

```bash
cd data
python preprocess.py
python pairs.py
cd ..
```

In `src/main.py`, set:

```python
mode = "train"
```

to train and save the best checkpoint, or:

```python
mode = "evaluate"
```

to load `best_model.pt` and run retrieval evaluation.

Then run:

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
- First run device: CPU

Validation loss was lowest around epoch 3 in the first 5-epoch run, so the evaluation uses the saved best checkpoint instead of the final epoch.

## Retrieval results

Test set size: 11,569 paired samples.

| Direction | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Molecule -> Phenotype | 0.0075 | 0.0265 | 0.0411 |
| Phenotype -> Molecule | 0.0066 | 0.0237 | 0.0392 |

These results are the first retrieval baseline and will be used as the reference point for later model and hyperparameter optimization.

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
- [x] Validation-based best-model saving
- [x] Test-set embedding generation
- [x] Bidirectional retrieval evaluation
- [x] Recall@1 / Recall@5 / Recall@10 baseline
- [ ] Training-speed and hyperparameter optimization
