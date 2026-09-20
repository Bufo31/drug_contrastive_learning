# False-Negative Analysis

## Goal

The vanilla contrastive loss treats every non-matching molecule-phenotype pair as a negative. This experiment checks whether some non-matching compounds nevertheless have very similar phenotype profiles, and whether the trained vanilla model also places those pairs closer in the shared embedding space.

The purpose is to test whether all negatives are equally reasonable negatives. This analysis motivates the next experiment; it does not by itself prove that a phenotype-aware loss will improve retrieval.

## Experimental design

- Randomly sample 5,000 compounds from the paired dataset with NumPy seed 31.
- L2-normalize the 737-dimensional phenotype vectors and calculate pairwise cosine similarity.
- Keep only the upper triangle so that each compound pair is counted once.
- Inspect the overall phenotype-similarity distribution and count highly similar non-matching pairs.
- Find the 10 phenotype-most-similar pairs as concrete examples.
- Load the tuned vanilla Dual Encoder and calculate the molecule-to-phenotype similarity matrix for the same 5,000 compounds.
- Use all non-matching molecule-phenotype pairs as the reference distribution for ordinary negatives.
- Average the two directions, Mol_i -> Ph_j and Mol_j -> Ph_i, for each compound pair.
- Compare mean model similarity at increasingly strict phenotype-similarity thresholds.

This is an exploratory analysis used to form a hypothesis. Final method comparison and test-set claims should remain separated from this exploratory stage.

## Results

### Phenotype similarity distribution

| Statistic | Value |
| --- | ---: |
| Mean | 0.01357 |
| Median | 0.01442 |
| 90th percentile | 0.27907 |
| 95th percentile | 0.35145 |
| 99th percentile | 0.48255 |

Among 12,497,500 unique pairs in the 5,000-sample subset:

| Threshold | Pair count | Approx. share |
| --- | ---: | ---: |
| similarity > 0.7 | 3,326 | 0.0266% |
| similarity > 0.8 | 605 | 0.00484% |
| similarity > 0.9 | 56 | 0.000448% |

Most non-matching phenotype pairs are not similar, but a small extreme tail of highly similar pairs exists.

### Top 10 phenotype-similar pairs

The model similarity below is the average of the two cross-modal directions.

| Rank | Compound A | Compound B | Phenotype similarity | Model similarity |
| ---: | --- | --- | ---: | ---: |
| 1 | JCP2022_046141 | JCP2022_054203 | 0.97463 | 0.62501 |
| 2 | JCP2022_054203 | JCP2022_067313 | 0.95746 | 0.70407 |
| 3 | JCP2022_012014 | JCP2022_078939 | 0.95534 | 0.44308 |
| 4 | JCP2022_046141 | JCP2022_067313 | 0.95463 | 0.68062 |
| 5 | JCP2022_097107 | JCP2022_066000 | 0.93582 | 0.53830 |
| 6 | JCP2022_115248 | JCP2022_019851 | 0.93573 | 0.49850 |
| 7 | JCP2022_089448 | JCP2022_028845 | 0.93501 | 0.59752 |
| 8 | JCP2022_009338 | JCP2022_039833 | 0.93496 | 0.58152 |
| 9 | JCP2022_115248 | JCP2022_079799 | 0.93265 | 0.45040 |
| 10 | JCP2022_009338 | JCP2022_019851 | 0.93044 | 0.62489 |

### Ordinary negative model similarity

| Statistic | Value |
| --- | ---: |
| Mean | 0.28674 |
| 95th percentile | 0.48387 |
| 99th percentile | 0.54563 |

Many of the highest phenotype-similarity pairs therefore have model similarities that are also unusually high relative to ordinary negatives.

### Phenotype similarity vs. model similarity

| Phenotype condition | Pair count | Mean model similarity |
| --- | ---: | ---: |
| > 0.3 | 1,036,318 | 0.33912 |
| > 0.5 | 96,853 | 0.37826 |
| > 0.7 | 3,326 | 0.47413 |
| > 0.8 | 605 | 0.51903 |
| > 0.9 | 56 | 0.56459 |

The threshold groups are nested rather than disjoint. As the phenotype-similarity threshold increases, the mean cross-modal model similarity also increases.

## Interpretation

The analysis supports three observations:

1. Highly similar phenotype profiles among different compounds are rare but clearly present.
2. The vanilla model already preserves part of this phenotype-similarity structure: phenotype-similar non-matching pairs tend to be closer in the learned molecule-phenotype embedding space than ordinary negatives.
3. The vanilla InfoNCE objective still labels every non-matching pair as a negative, regardless of phenotype similarity.

This suggests that negatives are heterogeneous. A very low-phenotype-similarity pair and a very high-phenotype-similarity pair may not deserve the same negative penalty.

The next hypothesis is therefore to test a simple phenotype-aware negative reweighting strategy, where phenotype-similar negatives receive weaker negative pressure. Improvement is not assumed in advance; it must be evaluated against the same vanilla baseline using controlled retrieval experiments.
