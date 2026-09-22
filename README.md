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
11. Analyze potential false negatives by comparing phenotype similarity with learned cross-modal similarity.
12. Test phenotype-aware negative weighting.
13. Test structure-aware negative weighting.
14. Compare the modified losses with the vanilla baseline across three random seeds.

## Project structure

```text
drug_contrastive_learning/
├── data/
│   ├── preprocess.py
│   ├── pairs.py
│   ├── raw/
│   └── processed/
├── experiments/
│   ├── baseline_tuning.md
│   ├── false_negative_analysis.md
│   └── similarity_aware_loss.md
├── src/
│   ├── data.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
├── false_negative.py
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

In the current `src/main.py`, set:

```text
mode = "1"   # train
mode = "2"   # evaluate
```

Then run:

```bash
python src/main.py
```

Run the false-negative analysis from the project root:

```bash
python false_negative.py
```

The training script was kept intentionally simple during experimentation. The active loss can be changed directly in `src/train.py`; the current uploaded version contains the vanilla, phenotype-aware, and structure-aware loss implementations and is set to the latest structure-aware experiment.

## Tuned vanilla baseline

After controlled hyperparameter experiments, the baseline configuration uses:

- Hidden dimension: 512
- Embedding dimension: 128
- Batch size: 256
- Learning rate: 0.001
- Weight decay: 0.0001
- Dropout: 0
- Temperature: 0.05
- Optimizer: Adam

Best-checkpoint retrieval for the selected single-run configuration:

| Direction | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Molecule -> Phenotype | 0.0112 | 0.0364 | 0.0558 |
| Phenotype -> Molecule | 0.0109 | 0.0333 | 0.0508 |

The full tuning history is recorded in [`experiments/baseline_tuning.md`](experiments/baseline_tuning.md).

## False-negative analysis

A fixed random sample of 5,000 compounds was used to examine whether all non-matching pairs should be treated as equally strong negatives.

Highly similar phenotype pairs were rare: 3,326 pairs had cosine similarity above 0.7, 605 above 0.8, and 56 above 0.9. At the same time, the mean learned cross-modal similarity increased as the phenotype-similarity threshold became stricter:

| Phenotype condition | Mean model similarity |
| --- | ---: |
| All non-matching negatives | 0.2867 |
| Phenotype similarity > 0.3 | 0.3391 |
| > 0.5 | 0.3783 |
| > 0.7 | 0.4741 |
| > 0.8 | 0.5190 |
| > 0.9 | 0.5646 |

The observation is that phenotype-similar non-matching pairs tend to remain closer in the learned embedding space than ordinary negatives. This motivated testing weaker negative pressure for similar pairs.

Full results are recorded in [`experiments/false_negative_analysis.md`](experiments/false_negative_analysis.md).

## Similarity-aware loss experiments

Two conservative negative-reweighting strategies were tested:

- **Phenotype-aware:** phenotype-similar non-matching pairs receive weaker negative pressure.
- **Structure-aware:** molecules with higher Morgan-fingerprint Tanimoto similarity receive weaker negative pressure.

Both use the same general idea:

```text
weight = 1 - alpha * similarity^gamma
```

with `alpha=0.25`, `gamma=2`, and the positive diagonal kept at weight 1.

Three-seed robustness results are summarized below. The values are mean +/- sample standard deviation.

| Metric | Vanilla | Phenotype-aware | Structure-aware |
| --- | ---: | ---: | ---: |
| Mol->Ph R@1 | 0.0098 +/- 0.0003 | 0.0096 +/- 0.0015 | 0.0104 +/- 0.0011 |
| Mol->Ph R@5 | 0.0334 +/- 0.0017 | 0.0333 +/- 0.0013 | 0.0330 +/- 0.0038 |
| Mol->Ph R@10 | 0.0539 +/- 0.0021 | 0.0531 +/- 0.0019 | 0.0520 +/- 0.0046 |
| Ph->Mol R@1 | 0.0102 +/- 0.0003 | 0.0109 +/- 0.0003 | 0.0098 +/- 0.0015 |
| Ph->Mol R@5 | 0.0333 +/- 0.0009 | 0.0325 +/- 0.0015 | 0.0315 +/- 0.0021 |
| Ph->Mol R@10 | 0.0515 +/- 0.0016 | 0.0502 +/- 0.0007 | 0.0498 +/- 0.0048 |

Neither modified loss produced a stable overall improvement over vanilla. Structure-aware weighting improved several metrics for some seeds, but it also increased seed-to-seed variance substantially.

The full experiment history, including the earlier soft-target attempt, alpha comparison, regularization test, multi-seed tables, and failure analysis, is recorded in [`experiments/similarity_aware_loss.md`](experiments/similarity_aware_loss.md).

## Failure analysis

The negative result suggests three main limitations:

1. **Similarity is not biological equivalence.** Phenotype similarity does not guarantee the same mechanism, and structural similarity does not guarantee the same phenotype or activity.
2. **Similarity preservation can conflict with exact-pair retrieval.** Recall@K rewards the unique matched molecule-phenotype pair, while a similar but non-matching sample is still an incorrect retrieval.
3. **False negatives may not be the main bottleneck.** Extremely high-similarity negatives are rare, and the vanilla model already preserves part of the similarity structure.

The main conclusion is that similarity alone is not sufficient to identify harmful false negatives.

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
- [x] False-negative analysis
- [x] Phenotype-aware contrastive loss
- [x] Structure-aware negative weighting
- [x] Three-seed robustness experiments
- [x] Failure analysis
