# Hetionet Table 5

`hetionet_table5_folds.csv` contains the fold-level test AUPRC and AUROC of
the final DEF-DTI predictions in the OptimusKG V3.2 revision. It includes ten
folds per regime. The source is the revision's `metrics_per_fold.csv` with
`arm=F`, the arm evaluated on `FINAL`. The public regime names correspond to
`warm_start`, `drug_coldstart`, and `protein_coldstart` in that artifact.
No pair identifiers, features, predictions, or comparator arms are included.

Source `metrics_per_fold.csv` SHA-256:
`a73bf8358ab451bc8c3d14fa7c7cdfe4b55779071b2fe4dab27d51ecdbe54aed`.
With the separate revision archive (`fits/`, `scores/`, and the source CSV),
`python scripts/verify_hetionet.py path/to/archive` refits the decision stage,
checks all 30 sets of final logits, and recomputes the test metrics.

| Regime | AUPRC (SD) | AUROC (SD) |
| --- | ---: | ---: |
| Warm-start | 0.842 (0.006) | 0.977 (0.001) |
| Compound-disjoint | 0.620 (0.036) | 0.897 (0.023) |
| Target-disjoint | 0.686 (0.020) | 0.938 (0.006) |

Run `defdti summarize-hetionet --fold-metrics results/hetionet_table5_folds.csv`
to generate unrounded statistics.
