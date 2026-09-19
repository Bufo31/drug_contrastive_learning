# Baseline Hyperparameter Tuning

This document records the controlled hyperparameter experiments used to choose the vanilla dual-encoder contrastive-learning baseline before the false-negative analysis stage.

## Final configuration

- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Dropout: `0`
- Temperature: `0.05`
- Batch size: `256`
- Hidden dimension: `512`
- Embedding dimension: `128`

The reported retrieval numbers come from the checkpoint with the lowest validation loss in each run. These are single-run comparisons; multi-seed robustness experiments are planned later.

## Learning rate

| Learning rate | Weight decay | Best validation loss | Best epoch |
| ---: | ---: | ---: | ---: |
| `1e-3` | `0` | 3.7006 | 3 |
| `3e-4` | `0` | 3.7872 | 2 |

Lowering the learning rate did not improve validation performance, so `1e-3` was retained.

## Weight decay

| Weight decay | Best validation loss | Best epoch |
| ---: | ---: | ---: |
| `0` | 3.7006 | 3 |
| `1e-5` | 3.6845 | 3 |
| `1e-4` | **3.6628** | 3 |

Moderate weight decay consistently improved the best validation loss, so `1e-4` was retained.

## Dropout

With `lr=1e-3` and `weight_decay=1e-4`, adding `dropout=0.1` produced a best validation loss of 3.6680, slightly worse than 3.6628 without dropout. The final baseline therefore uses no dropout.

## Temperature

| Temperature | Mol→Ph R@1 | Mol→Ph R@5 | Mol→Ph R@10 | Ph→Mol R@1 | Ph→Mol R@5 | Ph→Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.03` | 0.0093 | 0.0311 | 0.0515 | 0.0076 | 0.0309 | 0.0488 |
| `0.05` | 0.0095 | **0.0335** | **0.0531** | **0.0103** | **0.0320** | **0.0499** |
| `0.07` | **0.0104** | 0.0312 | 0.0526 | 0.0095 | 0.0301 | 0.0468 |
| `0.10` | 0.0101 | 0.0332 | 0.0500 | 0.0097 | 0.0314 | 0.0474 |

`0.05` gave the strongest overall retrieval results across the two directions.

## Batch size

| Batch size | Mol→Ph R@1 | Mol→Ph R@5 | Mol→Ph R@10 | Ph→Mol R@1 | Ph→Mol R@5 | Ph→Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | 0.0090 | 0.0299 | 0.0476 | 0.0080 | 0.0281 | 0.0462 |
| `128` | 0.0095 | 0.0335 | 0.0531 | 0.0103 | 0.0320 | 0.0499 |
| `256` | **0.0112** | **0.0364** | **0.0558** | **0.0109** | **0.0333** | **0.0508** |
| `512` | 0.0105 | 0.0362 | 0.0503 | 0.0105 | 0.0309 | 0.0478 |

Retrieval improved from batch size 64 to 256, then declined at 512. This may reflect saturation of useful in-batch negatives, more false negatives, or fewer optimizer updates under a fixed-epoch training budget; the current evidence does not distinguish among these explanations.

## Embedding dimension

| Embedding dim | Mol→Ph R@1 | Mol→Ph R@5 | Mol→Ph R@10 | Ph→Mol R@1 | Ph→Mol R@5 | Ph→Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128` | **0.0112** | **0.0364** | **0.0558** | **0.0109** | **0.0333** | **0.0508** |
| `256` | 0.0094 | 0.0296 | 0.0467 | 0.0084 | 0.0271 | 0.0416 |

Increasing the shared embedding to 256 dimensions reduced all six retrieval metrics.

## Hidden dimension

| Hidden dim | Mol→Ph R@1 | Mol→Ph R@5 | Mol→Ph R@10 | Ph→Mol R@1 | Ph→Mol R@5 | Ph→Mol R@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `256` | 0.0098 | 0.0297 | 0.0476 | 0.0080 | 0.0277 | 0.0430 |
| `512` | **0.0112** | **0.0364** | **0.0558** | **0.0109** | **0.0333** | **0.0508** |
| `1024` | 0.0080 | 0.0301 | 0.0481 | 0.0084 | 0.0289 | 0.0455 |

A hidden dimension of 512 performed best. The 1024-dimensional model fitted the training set more aggressively but generalized worse, while 256 reduced retrieval performance.

## Summary

The purpose of this stage was not exhaustive hyperparameter search. Each experiment changed one variable at a time to establish a reasonable vanilla contrastive-learning baseline before studying false negatives and modifying the loss.
