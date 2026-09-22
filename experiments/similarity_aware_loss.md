# Similarity-Aware Loss Experiments

This document records the loss-function experiments that followed the tuned vanilla baseline and false-negative analysis.

The goal was not to assume that similarity-aware losses must improve retrieval. The experiments were used to test whether some non-matching pairs should receive weaker negative pressure.

## 1. Starting point

The vanilla symmetric contrastive loss treats every off-diagonal molecule-phenotype pair as a negative.

The false-negative analysis showed that highly phenotype-similar non-matching pairs exist, although they are rare, and that the vanilla model already tends to place phenotype-similar pairs closer in the learned embedding space.

This motivated testing similarity-aware objectives.

## 2. Exploratory structure soft target

The first experiment used molecular Tanimoto similarity to build soft targets. Structure-similar compounds received part of the target probability instead of being treated as pure negatives.

Retrieval result:

| Direction | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Molecule -> Phenotype | 0.0099 | 0.0296 | 0.0462 |
| Phenotype -> Molecule | 0.0086 | 0.0274 | 0.0440 |

This was worse than the tuned vanilla baseline. The method was also conceptually strong because it partially treated structure-similar compounds as positives, even though structural similarity does not guarantee the same phenotype.

The later experiments therefore kept non-matching pairs as negatives and only reduced their negative weight.

## 3. Phenotype-aware negative weighting

For each batch, phenotype cosine similarity is calculated from the raw phenotype vectors.

For a non-matching pair with positive phenotype similarity s, the weight is

```text
w = 1 - alpha * s^gamma
```

The positive diagonal is kept at weight 1. The weight is applied by adding `log(w)` to the corresponding contrastive logit.

The main experiments used:

```text
temperature = 0.05
alpha = 0.25
gamma = 2
batch size = 256
learning rate = 1e-3
weight decay = 1e-4
epochs = 10
```

### Exploratory alpha comparison

| Alpha | Mol->Ph R@1 | Mol->Ph R@5 | Mol->Ph R@10 | Ph->Mol R@1 | Ph->Mol R@5 | Ph->Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.0112 | 0.0347 | 0.0541 | 0.0096 | 0.0335 | 0.0520 |
| 0.50 | 0.0097 | 0.0316 | 0.0526 | 0.0098 | 0.0316 | 0.0494 |
| 0.75 | 0.0109 | 0.0348 | 0.0556 | 0.0087 | 0.0322 | 0.0516 |

There was no consistent improvement across all six retrieval metrics. Alpha 0.25 was retained as the representative setting for the robustness experiment.

A stronger weight decay of `3e-4` was also tested with the phenotype-aware loss:

| Direction | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Molecule -> Phenotype | 0.0111 | 0.0328 | 0.0520 |
| Phenotype -> Molecule | 0.0098 | 0.0315 | 0.0509 |

It did not improve retrieval, so `1e-4` was retained.

## 4. Multi-seed robustness: Vanilla vs phenotype-aware

The train/validation/test split was kept fixed with NumPy seed 31. Only the PyTorch training seed was changed.

### Vanilla

| Seed | Mol->Ph R@1 | Mol->Ph R@5 | Mol->Ph R@10 | Ph->Mol R@1 | Ph->Mol R@5 | Ph->Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0099 | 0.0336 | 0.0537 | 0.0103 | 0.0335 | 0.0526 |
| 1 | 0.0100 | 0.0349 | 0.0561 | 0.0105 | 0.0341 | 0.0522 |
| 2 | 0.0095 | 0.0316 | 0.0520 | 0.0099 | 0.0324 | 0.0497 |

### Phenotype-aware

| Seed | Mol->Ph R@1 | Mol->Ph R@5 | Mol->Ph R@10 | Ph->Mol R@1 | Ph->Mol R@5 | Ph->Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0087 | 0.0329 | 0.0535 | 0.0108 | 0.0339 | 0.0508 |
| 1 | 0.0113 | 0.0347 | 0.0548 | 0.0112 | 0.0327 | 0.0505 |
| 2 | 0.0088 | 0.0322 | 0.0511 | 0.0106 | 0.0310 | 0.0494 |

Mean +/- sample standard deviation over three seeds:

| Metric | Vanilla | Phenotype-aware |
| --- | ---: | ---: |
| Mol->Ph R@1 | 0.0098 +/- 0.0003 | 0.0096 +/- 0.0015 |
| Mol->Ph R@5 | 0.0334 +/- 0.0017 | 0.0333 +/- 0.0013 |
| Mol->Ph R@10 | 0.0539 +/- 0.0021 | 0.0531 +/- 0.0019 |
| Ph->Mol R@1 | 0.0102 +/- 0.0003 | 0.0109 +/- 0.0003 |
| Ph->Mol R@5 | 0.0333 +/- 0.0009 | 0.0325 +/- 0.0015 |
| Ph->Mol R@10 | 0.0515 +/- 0.0016 | 0.0502 +/- 0.0007 |

The phenotype-aware loss did not produce a stable overall retrieval improvement.

## 5. Structure-aware negative weighting

A second hypothesis used molecular structure instead of phenotype similarity.

Tanimoto similarity is calculated from Morgan fingerprints, and structure-similar non-matching compounds receive weaker negative pressure:

```text
w = 1 - alpha * Tanimoto^gamma
```

The representative setting again used:

```text
temperature = 0.05
alpha = 0.25
gamma = 2
batch size = 256
learning rate = 1e-3
weight decay = 1e-4
epochs = 10
```

### Multi-seed results

| Seed | Mol->Ph R@1 | Mol->Ph R@5 | Mol->Ph R@10 | Ph->Mol R@1 | Ph->Mol R@5 | Ph->Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0107 | 0.0354 | 0.0539 | 0.0115 | 0.0334 | 0.0525 |
| 1 | 0.0113 | 0.0350 | 0.0553 | 0.0091 | 0.0318 | 0.0526 |
| 2 | 0.0091 | 0.0286 | 0.0468 | 0.0087 | 0.0293 | 0.0442 |

Mean +/- sample standard deviation:

| Metric | Vanilla | Structure-aware |
| --- | ---: | ---: |
| Mol->Ph R@1 | 0.0098 +/- 0.0003 | 0.0104 +/- 0.0011 |
| Mol->Ph R@5 | 0.0334 +/- 0.0017 | 0.0330 +/- 0.0038 |
| Mol->Ph R@10 | 0.0539 +/- 0.0021 | 0.0520 +/- 0.0046 |
| Ph->Mol R@1 | 0.0102 +/- 0.0003 | 0.0098 +/- 0.0015 |
| Ph->Mol R@5 | 0.0333 +/- 0.0009 | 0.0315 +/- 0.0021 |
| Ph->Mol R@10 | 0.0515 +/- 0.0016 | 0.0498 +/- 0.0048 |

Structure-aware weighting improved several metrics in seed 0 and seed 1, but seed 2 degraded strongly. The average improvement was limited to Mol->Ph Recall@1, while the other five mean metrics were lower than vanilla. The variance was also substantially larger.

Therefore the structure-aware result is not a robust improvement.

## 6. Failure analysis

The experiments suggest three main explanations.

### 6.1 Similarity is not biological equivalence

Phenotype similarity does not necessarily imply the same mechanism of action, and molecular structural similarity does not guarantee the same phenotype or activity.

Therefore

```text
high similarity
does not necessarily mean
this pair should be a weaker negative
```

Reducing the penalty for every similar pair may also weaken useful discriminative negatives.

### 6.2 Similarity preservation can conflict with exact-pair retrieval

The evaluation task requires exact one-to-one retrieval:

```text
Mol_i <-> Ph_i
```

A similar but non-matching sample is still counted as an incorrect retrieval result. A loss that allows similar non-matching samples to remain closer may create a smoother representation without improving exact Recall@K.

### 6.3 False negatives may not be the main bottleneck

The false-negative analysis showed that extremely high phenotype-similarity negatives exist, but they are rare.

The vanilla model also already preserves part of the similarity structure: phenotype-similar non-matching pairs tend to have higher learned cross-modal similarity than ordinary negatives.

Therefore changing the whole contrastive objective to protect a small subset of possible false negatives may provide less benefit than expected.

## 7. Conclusion

The experiments support the observation that contrastive negatives are heterogeneous, but they do not support a simple rule that all phenotype-similar or structure-similar negatives should be uniformly down-weighted.

The main conclusion is:

```text
similarity alone is insufficient to identify harmful false negatives
```

A stronger future direction would be to distinguish true false negatives from useful hard negatives using richer biological or mechanistic information, rather than relying on a single similarity score.
