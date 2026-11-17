# DEF-DTI

**Decision-Level Evidence Factorization for Drug-Target Interaction Prediction Across Data Regimes**

DEF-DTI combines a frozen hierarchical cross-attention (HCA) pair predictor,
label-neighborhood support derived from training interactions, and a
coverage-masked additive residual from knowledge-graph embedding (KGE) scores.
The revised manuscript uses OptimusKG V3.2 for KGE on all seven benchmarks.

This package includes the pair-only HCA inference path, decision-level fitting
and scoring for the revised Hetionet benchmark, and the fold metrics for its
Table 5. The previous `scrdti` interface remains available for older checkouts
and checkpoints.

## Install and verify

Use Python 3.10 or 3.11 and a PyTorch build appropriate for your platform.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,chem]"
defdti smoke
defdti summarize-hetionet --fold-metrics results/hetionet_table5_folds.csv
python -m pytest -q
```

The smoke test and Table 5 aggregation run on CPU without downloading models
or benchmark data.

## Reproduce the Hetionet decision stage

See [the input schemas, fitting commands, and regime rules](docs/DEF_DTI.md).
Start from saved base logits and KGE scores to run the decision stage, or use
the Python API to compute HCA logits and refit the base from prepared inputs.
Each of the three regimes has ten folds. In the target-disjoint regime,
the reported final prediction is the base logit.

Exact reproduction from raw inputs also needs the original folds, feature
stores, fitted HCA pair predictors, label-neighborhood features, and OptimusKG
inputs described in the workflow guide. These artifacts are not bundled.
The supplied metrics and commands cover the DEF-DTI row of Table 5, not
end-to-end replay of all seven benchmarks or the manuscript figures.

## Use the previous release

The [SCR-DTI v0.1 guide](docs/LEGACY_SCR_DTI.md) documents the `scrdti` CLI,
original configurations, and checkpoint compatibility. Its graph-context head
and bounded residual describe the previous release.

See [CITATION.cff](CITATION.cff) for citation details and [NOTICE](NOTICE) for
third-party code and data terms.
