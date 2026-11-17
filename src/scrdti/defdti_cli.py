"""Command-line entry points for the DEF-DTI revision."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scrdti.defdti import REGIMES, Residual, aggregate_hetionet, fit_hetionet_residual, predict

INPUTS = ["base_logit", "kge_score", "covered"]
IDS = ["molecule_id", "protein_id"]
GRAPH = "OptimusKG V3.2"


def read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={c: str for c in IDS})
    raise ValueError("Use .csv or .parquet")


def pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    ids = frame[IDS]
    if ids.isna().any().any():
        raise ValueError("Pair identifiers must be present and unique")
    ids = ids.astype(str).apply(lambda column: column.str.strip())
    if ids.eq("").any().any() or ids.duplicated().any() or ids.empty:
        raise ValueError("Pair identifiers must be present and unique")
    return set(ids.itertuples(index=False, name=None))


def scoring_frame(path: Path) -> pd.DataFrame:
    frame = read(path)
    if set(frame.columns) != set(IDS + INPUTS):
        raise ValueError("Scoring input must contain only the pair IDs and predictors, without labels")
    return frame


def fit(args) -> None:
    frame = read(args.valid)
    if set(frame.columns) != set(IDS + INPUTS + ["label"]):
        raise ValueError("Validation input must contain the pair IDs, predictors and label")
    validation_pairs = pairs(frame)
    if validation_pairs & pairs(scoring_frame(args.score_pairs)):
        raise ValueError("Validation and scoring pairs overlap")
    params = (
        fit_hetionet_residual(frame.base_logit, frame.label, frame.kge_score, frame.covered)
        if args.regime != "target-disjoint"
        else Residual()
    )
    result = {"graph": GRAPH, "regime": args.regime, "residual": asdict(params)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def score(args) -> None:
    frame = scoring_frame(args.inputs)
    pairs(frame)
    params = json.loads(args.params.read_text(encoding="utf-8"))
    if params["graph"] != GRAPH:
        raise ValueError(f"The revision uses {GRAPH}")
    result = predict(
        *(frame[c].to_numpy() for c in INPUTS),
        Residual(**params["residual"]), params["regime"],
    )
    frame[IDS].assign(**result).to_csv(args.output, index=False)


def smoke(_args) -> None:
    base = np.array([-1.0, 0.0, 1.0])
    out = predict(base, [2.0, np.nan, 4.0], [1, 0, 1], Residual(w=0.5, active=True), "warm-start")
    np.testing.assert_array_equal(out["logit"], [0.0, 0.0, 3.0])
    print("DEF-DTI CPU smoke test passed")


def summary(args) -> None:
    result = aggregate_hetionet(read(args.fold_metrics)).to_csv(index=False)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


def main() -> None:
    parser = argparse.ArgumentParser(prog="defdti")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="run a CPU check with synthetic inputs").set_defaults(func=smoke)
    command = sub.add_parser("fit-residual", help="fit the Hetionet residual on validation data")
    command.add_argument("--valid", type=Path, required=True)
    command.add_argument(
        "--score-pairs", type=Path, required=True, help="label-free scoring input for the overlap check"
    )
    command.add_argument("--regime", choices=REGIMES, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=fit)
    command = sub.add_parser("predict", help="score frozen inputs without reading test labels")
    command.add_argument("--inputs", type=Path, required=True)
    command.add_argument("--params", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=score)
    command = sub.add_parser("summarize-hetionet", help="aggregate fold metrics for Table 5")
    command.add_argument("--fold-metrics", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command.set_defaults(func=summary)
    args = parser.parse_args()
    try:
        args.func(args)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
