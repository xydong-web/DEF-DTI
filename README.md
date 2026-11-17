# DEF-DTI

**Decision-Level Evidence Factorization for Drug-Target Interaction Prediction Across Data Regimes**

DEF-DTI predicts whether a compound interacts with a protein target. It separates
three sources of evidence: an HCA pair predictor built from compound and protein
representations, neighborhood support derived from labeled training pairs, and
an additive knowledge-graph embedding (KGE) residual applied to covered pairs.
The manuscript uses OptimusKG V3.2 for its graph evidence.

| Goal | Start here | Inputs |
| --- | --- | --- |
| Check your installation | [Run the CPU example](#run-the-cpu-example) | None; the data are generated locally |
| Train and test a pair model | [Download data](#download-the-benchmark-data) and [train on Davis](#train-and-test-on-davis) | Binary pair labels, compound SMILES, protein sequences, and partition assignments |
| Fit or score the revised Hetionet decision | [Run the Hetionet decision stage](#run-the-hetionet-decision-stage) | Fold-specific HCA, neighborhood, and KGE inputs |
| Check the Table 5 numbers | [Recompute Table 5](#recompute-table-5) | The 30 fold metrics included here |

Benchmark data, pretrained encoder weights, and trained DEF-DTI checkpoints
are obtained separately. The synthetic example and Table 5 summary need none
of these downloads.

## Install

Use Python 3.10 or 3.11. Run the commands from the repository root. The examples
use Bash; in PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`
and put multiline commands on one line instead of using Bash's `\` continuations.

```bash
git clone https://github.com/xydong-web/DEF-DTI.git
cd DEF-DTI
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,chem]"
```

If you plan to start from SMILES and sequences, add the encoder dependencies:

```bash
python -m pip install -e ".[dev,chem,encoders]"
```

The encoder extra requires PyTorch 2.6 to read the published MolT5 checkpoint.
Use a PyTorch 2.6 build appropriate for your CPU or CUDA environment. The
encoder weights are downloaded on the first extraction run; training from an
existing feature archive does not need this extra.

## Run the CPU example

This example exercises both the decision-stage smoke check and a complete
pair-model train/test run. The generated features and scores are synthetic.

```bash
defdti smoke
defdti make-synthetic --output runs/demo/pairs.npz --n 96 --seed 7
defdti train --config configs/examples/tiny_cpu.yaml \
  --data runs/demo/pairs.npz --output runs/demo/checkpoint.pt --device cpu
defdti predict-pairs --config configs/examples/tiny_cpu.yaml \
  --data runs/demo/pairs.npz --checkpoint runs/demo/checkpoint.pt \
  --output runs/demo/test_predictions.csv --split 2 --device cpu
defdti evaluate --predictions runs/demo/test_predictions.csv
python -m pytest -q
```

Look for `DEF-DTI CPU smoke test passed`, a checkpoint and prediction file
in `runs/demo/`, and a JSON report with `auprc` and `auroc` values. Tests
requiring encoder dependencies or separate historical research artifacts
are skipped when those inputs are absent.

## Download the benchmark data

Use the binary labels in the source release. DrugBank, Davis, and KIBA use
fixed 80%/10%/10% pairwise-random partitions in the manuscript. Start with the
EviDTI binary tables rather than applying a new threshold to raw affinity
measurements. The other four benchmarks use the GRAM-DTI/DTIAM data and
protocol: one labeled positive to ten sampled unobserved pairs. A sampled
negative is a benchmark label, not experimental proof of non-interaction.

| Benchmark | Source and required files | What the positive label means | Manuscript split |
| --- | --- | --- | --- |
| DrugBank | [EviDTI benchmark][evidti] → `drugbank_dataset/total_cid_unid_csv.csv` and `metadata.csv`; join `cid` and `Uniprot ID` to `cid` and `uid` for structures and sequences. For raw DrugBank records, see [DrugBank releases][drugbank]. | A recorded DrugBank interaction in the curated binary benchmark. | Fixed 80/10/10 pairwise-random |
| Davis | [EviDTI benchmark][evidti] → `davis_dataset/davis_total_cid_unid.csv` with `cid`, `uid`, `smiles`, `seq`, and `label`. See the [original Davis study][davis]. | A positive kinase–inhibitor pair under the EviDTI binary encoding of the Davis affinity data. | Fixed 80/10/10 pairwise-random |
| KIBA | [EviDTI benchmark][evidti] → `KIBA_dataset/KIBA_total_cid_unid.csv` with `cid`, `uid`, `smiles`, `seq`, and `label`. See the [original KIBA study][kiba]. | A positive pair under the EviDTI binary encoding of the integrated kinase bioactivity scores. | Fixed 80/10/10 pairwise-random |
| Yamanishi 08 | [GRAM-DTI downstream files][gram] → `yamanishi_08/{dti,drug_smiles,protein_seq}.csv`. See the [original benchmark study][yamanishi]. | An annotated interaction from the Yamanishi 08 benchmark. | 10 folds per entity-aware regime |
| Hetionet | [GRAM-DTI downstream files][gram] → `Hetionet/{dti,drug_smiles,protein_seq}.csv`. See [Hetionet v1.0][hetionet] for the source graph and relation schema. | An annotated compound–target interaction in the downstream benchmark. | 10 folds per entity-aware regime |
| Activation | [GRAM-DTI downstream files][gram] → `activation/{dti,drug_smi,tar_seq}.csv`. The upstream mechanism labels are derived from the [Therapeutic Target Database][ttd] via DTIAM. | A recorded activation of the target by the compound. | 5 folds per entity-aware regime |
| Inhibition | [GRAM-DTI downstream files][gram] → `inhibition/{dti,drug_smi,tar_seq}.csv`. The upstream mechanism labels are derived from the [Therapeutic Target Database][ttd] via DTIAM. | A recorded inhibition of the target by the compound. | 5 folds per entity-aware regime |

The three GRAM-DTI files listed for each dataset are tab-separated despite
the `.csv` suffix. `dti.csv` lists positive pairs by ID; the other two files
map IDs to compound SMILES and protein sequences. [The upstream split
generator][gram-splits] produces train/test folds. To run a new experiment,
reserve validation data from each training fold using that same regime.
For exact manuscript replay, use the archived pair labels and fold manifests;
the public source tables and newly generated partitions are not substitutes.

| Regime | What is held out |
| --- | --- |
| Pairwise-random | Pairs are partitioned; compounds or targets may recur across splits. |
| Warm-start | Pairs are held out without holding out whole compound or target groups. Use the chosen benchmark's warm-start manifest. |
| Compound-disjoint | Test compounds are unseen in training; targets may recur. |
| Target-disjoint | Test targets are unseen in training; compounds may recur. |

Keep source files under `data/raw/`, prepared CSV/NPZ files under
`data/prepared/`, and run outputs under `runs/`. Record the source revision,
checksums, license, negative sampling recipe, and split seed or manifest.
The [data guide](docs/DATA.md) describes optional feature arrays and graph
controls. For graph experiments also obtain `nodes.parquet` and
`edges.parquet` from [OptimusKG V3.2][optimuskg], along with the entity
mappings used for the fold. This is a separate download from the Hetionet
interaction benchmark. Raw DrugBank downloads are subject to DrugBank's
access and licensing terms.

## Train and test on Davis

### 1. Prepare the pair CSV

Download the Davis table from the source above as
`data/raw/davis_total_cid_unid.csv`. Save and run the following Python code
from the repository root. It creates a **new**, stratified 80/10/10 split
with seed 42 to demonstrate a real-data run. To reproduce the paper split,
use the archived assignments keyed by the same pair IDs.

```python
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

pairs = pd.read_csv("data/raw/davis_total_cid_unid.csv", dtype={"cid": str, "uid": str})
pairs = pairs.rename(columns={"cid": "molecule_id", "uid": "protein_id", "seq": "protein_sequence"})
ids = ["molecule_id", "protein_id"]
pairs = pairs[ids + ["smiles", "protein_sequence", "label"]].drop_duplicates()
if pairs.isna().any().any() or pairs.duplicated(ids).any() or set(pairs.label) != {0, 1}:
    raise ValueError("Expected one complete row per pair and both binary labels")
pairs = pairs.sort_values(ids).reset_index(drop=True)
train, held_out = train_test_split(pairs, test_size=0.2, random_state=42, stratify=pairs.label)
valid, test = train_test_split(held_out, test_size=0.5, random_state=42, stratify=held_out.label)
pairs["split"] = 0
pairs.loc[valid.index, "split"] = 1
pairs.loc[test.index, "split"] = 2
output = Path("data/prepared/davis_pairs.csv")
output.parent.mkdir(parents=True, exist_ok=True)
pairs.to_csv(output, index=False)
```

The feature extractor reads `smiles`, `protein_sequence`, `label`, and `split`.
Use `split=0` for training, `1` for validation, and `2` for test. Keep both
ID columns and a stable row order. The pair-model prediction CSV does not
include IDs; it follows the original row order within the selected split.
For other benchmarks, resolve duplicate or missing source IDs before
splitting, keep each pair in one partition, and hold out whole entities
for the disjoint regimes.

### 2. Extract frozen features

| Input view | Default checkpoint | Default pooled width |
| --- | --- | ---: |
| Canonical isomeric SMILES | [MoLFormer-XL-both-10pct][molformer] | 768 |
| `SMILES=` followed by the standardized SMILES | [MolT5-base][molt5] encoder | 768 |
| Protein amino-acid sequence | [ESM-2 650M][esm] | 1,280 |

With `.[encoders,chem]` installed, run:

```bash
defdti precompute --input data/prepared/davis_pairs.csv \
  --output data/prepared/davis_pairs.npz --batch-size 8 --device cuda
```

Use `--device cpu` without CUDA, or reduce `--batch-size` if extraction runs
out of memory. The extractor standardizes the SMILES with RDKit and averages
encoder token states using the attention mask. It writes `smiles`, `text`,
`protein`, `labels`, `split`, and `has_text` arrays in input row order.
Graph support features, KGE scores, and coverage masks are prepared separately.
If you substitute encoders, match `smiles_input_dim`, `text_input_dim`, and
`protein_input_dim` in the training config to the saved feature widths.
For exact replay, use the archived feature store and encoder revisions.

### 3. Train, predict, and evaluate

```bash
defdti train --config configs/paper/davis.yaml \
  --data data/prepared/davis_pairs.npz --output runs/davis/checkpoint.pt --device cuda
defdti predict-pairs --config configs/paper/davis.yaml \
  --data data/prepared/davis_pairs.npz --checkpoint runs/davis/checkpoint.pt \
  --output runs/davis/test_predictions.csv --split 2 --device cuda
defdti evaluate --predictions runs/davis/test_predictions.csv
```

Use `--device cpu` when working without CUDA. `train` selects the checkpoint
with the best validation AUPRC; `predict-pairs` loads it and scores the
requested split. Use `--split 1` (with a separate output filename) to export
validation predictions for threshold selection or calibration. Keep that
selection off the test set. `evaluate` reports average precision (AUPRC),
AUROC, thresholded metrics at 0.5, and probability calibration metrics.
Retain the configuration, random seed, feature revisions, checkpoint, and
predictions with each run.

The [historical pair-model templates](configs/paper/README.md) show the
configuration layout for all seven datasets. The intramodel graph branch
in those templates is separate from the revised decision stage below.

## Run the Hetionet decision stage

The [Hetionet workflow](docs/DEF_DTI.md) shows how to compute pair logits and
fit the base combiner. For each fold:

1. Use the fold's frozen HCA checkpoints (seeds 42, 43, and 44) with their
   matching pooled feature stores, and average the **logits** across seeds.
2. Compute neighborhood support from training labels only. Fit the base
   combiner on validation rows using the HCA logits and the fold's support
   channels (`g0`–`g15` for warm-start; support logit `eh` for disjoint
   regimes). Export `base_logit` for validation and test pairs.
3. Join fold-aligned KGE head scores and coverage from OptimusKG V3.2.
   Coverage requires both endpoints to map to the graph. The embedding
   graph excludes direct compound–protein relations and their inverses;
   the Hetionet graph-derived neighborhood kernels also exclude compound–gene
   relations.

Create `test.parquet` with exactly `molecule_id`, `protein_id`, `base_logit`,
`kge_score`, and `covered`. `valid.parquet` has the same columns plus `label`.
Pair IDs must be unique and validation pairs must differ from scoring pairs.
From the directory containing the two files:

```bash
defdti fit-residual --regime warm-start --valid valid.parquet \
  --score-pairs test.parquet --output residual.json
defdti predict --inputs test.parquet --params residual.json --output predictions.csv
```

Repeat per fold and set `--regime` to `warm-start`, `compound-disjoint`, or
`target-disjoint`. Warm-start and compound-disjoint fit a validation
residual for covered rows when the coverage and class-count checks pass.
Target-disjoint uses the base logit alone. `predict` scores the frozen
decision inputs without test labels; `predict-pairs` scores a feature NPZ
with a pair-model checkpoint.

After scoring, join the separately saved test labels by the two IDs. Verify
that the join is one-to-one and covers every test pair, then compute
`average_precision_score` and `roc_auc_score` on the final `logit`.

## Recompute Table 5

```bash
defdti summarize-hetionet --fold-metrics results/hetionet_table5_folds.csv
```

This recomputes the mean and sample standard deviation of the DEF-DTI row
from 30 included test-fold measurements:

| Regime | AUPRC (SD) | AUROC (SD) |
| --- | ---: | ---: |
| Warm-start | 0.842 (0.006) | 0.977 (0.001) |
| Compound-disjoint | 0.620 (0.036) | 0.897 (0.023) |
| Target-disjoint | 0.686 (0.020) | 0.938 (0.006) |

To verify individual predictions, obtain the separate revision archive
containing `fits/`, `scores/`, and `metrics_per_fold.csv`. Place its
contents under `data/revision/` and run:

```bash
python scripts/verify_hetionet.py data/revision
```

See the [results notes](results/README.md) for provenance and the source
checksum. End-to-end manuscript replay additionally requires the original
folds, checkpoint and feature stores (including the archived MoLFormer
features), graph inputs, and upstream preparation scripts. The included
metrics, decision code, and historical pair-model templates do not by
themselves reproduce the other tables or manuscript figures.

## Further documentation

- [Data and representations](docs/DATA.md): array schemas and graph controls.
- [Pair-model workflow](docs/PAIR_MODEL_WORKFLOW.md): training, calibration, and checkpoint compatibility.
- [Reproducibility notes](docs/REPRODUCIBILITY.md) and [architecture](docs/ARCHITECTURE.md).
- [Citation](CITATION.cff), [license](LICENSE), and [third-party notices](NOTICE).

[evidti]: https://github.com/zhaoyanpeng208/EviDTI/tree/8734bd7278edb2453c76d3072f292fa19f152802/dataset
[drugbank]: https://go.drugbank.com/releases/latest
[davis]: https://doi.org/10.1038/nbt.1990
[kiba]: https://doi.org/10.1021/ci400709d
[gram]: https://huggingface.co/datasets/FengJiang02/GRAM-DTI/tree/main/downstream
[gram-splits]: https://huggingface.co/datasets/FengJiang02/GRAM-DTI/blob/main/downstream/make_splits.py
[yamanishi]: https://doi.org/10.1093/bioinformatics/btn162
[hetionet]: https://github.com/hetio/hetionet
[ttd]: https://ttd.idrblab.cn/
[optimuskg]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FIYNGEV&version=3.2
[molformer]: https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct
[molt5]: https://huggingface.co/laituan245/molt5-base
[esm]: https://huggingface.co/facebook/esm2_t33_650M_UR50D
