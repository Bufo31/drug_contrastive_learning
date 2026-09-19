# Drug Contrastive Learning

A small PyTorch project for learning a shared embedding space between molecular structures and cell-phenotype profiles with a dual-encoder contrastive-learning model.

This repository records the project step by step so that the baseline, evaluation pipeline, tuning process, and later research experiments remain visible in Git history.

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
10. Tune the vanilla baseline with controlled one-variable-at-a-time experiments.

## Project structure

```text
drug_contrastive_learning/
├── data/
│   ├── preprocess.py
│   ├── pairs.py
│   ├── raw/
│   └── processed/
├── experiments/
│   └── baseline_tuning.md
├── src/
│   ├── data.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
├── test.py
├── parquet_report.txt
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

In `src/main.py`, set `mode = "train"` to train and save the best checkpoint, or `mode = "evaluate"` to load `best_model.pt` and run retrieval evaluation.

Then run:

```bash
python src/main.py
```

## Tuned vanilla baseline

After controlled hyperparameter experiments, the current baseline uses:

- Hidden dimension: 512
- Embedding dimension: 128
- Batch size: 256
- Learning rate: 0.001
- Weight decay: 0.0001
- Dropout: 0
- Temperature: 0.05
- Optimizer: Adam
- Epochs: 20 with best-checkpoint selection by validation loss

Best-checkpoint retrieval for the selected configuration:

| Direction | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Molecule -> Phenotype | 0.0112 | 0.0364 | 0.0558 |
| Phenotype -> Molecule | 0.0109 | 0.0333 | 0.0508 |

The full tuning history is recorded in [`experiments/baseline_tuning.md`](experiments/baseline_tuning.md). These are single-run tuning results; multi-seed robustness tests are intentionally reserved for a later stage.

## Dependencies

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
- [x] Bidirectional retrieval evaluation
- [x] Recall@1 / Recall@5 / Recall@10 baseline
- [x] Controlled vanilla-baseline hyperparameter tuning
- [ ] False-negative analysis
- [ ] Phenotype-aware contrastive loss
- [ ] Ablation and robustness experiments
