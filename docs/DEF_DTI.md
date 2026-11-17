# DEF-DTI Hetionet workflow

## Inputs and scope

The decision stage starts from a frozen pair prediction and separately prepared
support. For each fold, `z` is the mean pair logit from HCA training seeds 42,
43, and 44. The warm-start base combiner uses `z` and the 16 support channels
`g0` through `g15`. The compound- and target-disjoint combiners use `z` and
`eh`, the support logit of a histogram gradient-boosting model fitted on
training features. The combiner is a standard scaler followed by logistic
regression (`C=0.5`), fitted on validation rows.

The final score adds a linear residual to the fixed base logit on rows
covered by the graph. The residual is fitted with logistic loss on validation
rows only. Standardization uses the mean and population standard deviation
of covered validation scores, with `1e-6` added to the scale. The fit requires
at least 200 covered rows and at least 10 examples of each class among those
rows.

| Regime | Table 5 final logit |
| --- | --- |
| `warm-start` | Base plus coverage-masked KGE residual |
| `compound-disjoint` | Base plus coverage-masked KGE residual |
| `target-disjoint` | Base, with no residual |

The manuscript pools frozen MoLFormer, MolT5 and ESM-2 token representations
before training the HCA pair predictors. Exact replay requires the matching
feature store, in particular the archived MoLFormer store, and the
corresponding checkpoints and fold definitions. The KGE scores use
OptimusKG V3.2 (Harvard Dataverse DOI `10.7910/DVN/IYNGEV`) and fold-specific
pair scoring heads. The predictive embedding view excludes direct
compound-protein relations, including inverses; the graph-derived Hetionet
neighborhood features also exclude compound-gene relations. Coverage requires
both endpoints to map to graph nodes. The package expects the resulting scores
and masks as inputs; it does not build their exact upstream artifacts.

## Obtain the frozen pair logit

Given the trained pair-only state dictionaries and the matching pooled
embeddings, run the models on aligned batches with PyTorch:

```python
import torch
from scrdti.defdti_pair import load_pair_predictor

models = [load_pair_predictor(state) for state in (state42, state43, state44)]
with torch.inference_mode():
    z = torch.stack([
        model(smiles_emb, text_emb, protein_emb, has_text=has_text).pair_logit
        for model in models
    ]).mean(dim=0)
```

For an original trusted research checkpoint, pass its `model` state
dictionary. The loader checks all active HCA keys and tensor shapes, and
does not read serialized files itself. Inputs are the two pooled molecular
views and the pooled protein sequence view, in that order. Match the
training-time feature store and molecular view mask. This interface has no
graph-context head or bounded residual.

## Fit the base combiner

Supply the correctly prepared `valid` and label-free `test` DataFrames for
one fold. The required validation label column is `label`. Keep pair IDs and
fold membership unchanged when exporting scores.

```python
from scrdti.defdti import fit_base

columns = ["z"] + [f"g{i}" for i in range(16)]  # warm-start
# For compound-disjoint or target-disjoint, use ["z", "eh"].
model = fit_base(valid, columns)
valid["base_logit"] = model.decision_function(valid[columns].to_numpy())
test["base_logit"] = model.decision_function(test[columns].to_numpy())
```

## Fit and score from frozen inputs

Both files can be CSV or Parquet. `test.parquet` has exactly the five columns
below. `valid.parquet` has those columns plus the binary `label` column. Pair
IDs must identify each row uniquely and the validation and scoring pairs
must be disjoint. In saved research exports, `b` is `base_logit` and
`kge_logit` is `kge_score`.

| Column | Meaning |
| --- | --- |
| `molecule_id`, `protein_id` | Stable compound and protein identifiers |
| `base_logit` | Finite output of the frozen base combiner |
| `kge_score` | Finite KGE head score on covered rows |
| `covered` | `0`/`1` or Boolean pair coverage |

```bash
defdti fit-residual --regime warm-start --valid valid.parquet \
  --score-pairs test.parquet --output residual.json
defdti predict --inputs test.parquet --params residual.json --output predictions.csv
```

Run the fit separately per fold. The score file contains IDs, base logit,
residual, final logit and its sigmoid probability. Join test labels by pair
only after scoring. Compute AUPRC with `average_precision_score` and AUROC
with `roc_auc_score`, both on final logits.

## Recompute the Table 5 summary

```bash
defdti summarize-hetionet --fold-metrics results/hetionet_table5_folds.csv
```

The command validates ten folds per regime and computes the mean and sample
standard deviation. See [the results notes](../results/README.md) for
provenance and the printed Table 5 values. The three pairwise-random
benchmarks and the remaining entity-aware benchmarks use separate split and
support recipes. The legacy `configs/paper/` templates do not reproduce those
revised results.
