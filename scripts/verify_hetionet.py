"""Verify Table 5 from separately obtained revision artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scrdti.defdti import Residual, aggregate_hetionet, fit_base, fit_hetionet_residual, predict

REGIMES = {
    "warm_start": "warm-start",
    "drug_coldstart": "compound-disjoint",
    "protein_coldstart": "target-disjoint",
}
IDS = ["molecule_id", "protein_id"]
FROZEN = ["b", "kge_logit", "covered"]
METRICS = ["auprc", "auroc"]


def verify(root: Path) -> pd.DataFrame:
    source = pd.read_csv(root / "metrics_per_fold.csv", float_precision="round_trip")
    final = source.loc[source.arm == "F", ["regime", "fold", *METRICS]].copy()
    final["regime"] = final.regime.map(REGIMES)
    summary = aggregate_hetionet(final)
    reported = final.set_index(["regime", "fold"])
    for archived, regime in REGIMES.items():
        columns = ["z", *[f"g{i}" for i in range(16)]] if archived == "warm_start" else ["z", "eh"]
        for fold in range(10):
            fit = root / "fits" / f"{archived}_fold{fold}"
            valid = pd.read_parquet(
                fit / "valid_features.parquet", columns=[*IDS, "y", *FROZEN, *columns]
            ).rename(columns={"y": "label"})
            test = pd.read_parquet(fit / "test_features.parquet", columns=[*IDS, *FROZEN, *columns])
            if not valid.merge(test[IDS], on=IDS, validate="one_to_one").empty:
                raise ValueError(f"Validation and scoring pairs overlap: {regime} fold {fold}")
            model = fit_base(valid, columns)
            for frame in (valid, test):
                np.testing.assert_allclose(
                    model.decision_function(frame[columns].to_numpy()), frame.b, rtol=0, atol=1e-8
                )
            saved = Residual(**json.loads((fit / "residual_parameters.json").read_text()))
            refit = fit_hetionet_residual(valid.b, valid.label, valid.kge_logit, valid.covered)
            np.testing.assert_allclose(
                [refit.mu, refit.sd, refit.w, refit.bias, refit.active],
                [saved.mu, saved.sd, saved.w, saved.bias, saved.active],
                rtol=0, atol=1e-8,
            )
            out = predict(test.b, test.kge_logit, test.covered, refit, regime)
            expected = pd.read_parquet(
                root / "scores" / f"hetionet__{archived}__fold{fold}.parquet",
                columns=[*IDS, "label", "FINAL"], filters=[("split", "==", "test")],
            )
            joined = test[IDS].assign(logit=out["logit"]).merge(
                expected, on=IDS, how="outer", validate="one_to_one", indicator=True
            )
            if not joined["_merge"].eq("both").all():
                raise ValueError(f"Score IDs do not match: {regime} fold {fold}")
            np.testing.assert_allclose(joined.logit, joined.FINAL, rtol=0, atol=1e-8)
            metrics = [
                average_precision_score(joined.label, joined.logit),
                roc_auc_score(joined.label, joined.logit),
            ]
            np.testing.assert_allclose(metrics, reported.loc[regime, fold], rtol=0, atol=1e-12)
        print(f"{regime}: 10 folds verified", flush=True)
    print(summary.to_csv(index=False), end="")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, help="export verified fold metrics")
    args = parser.parse_args()
    folds = verify(args.archive)
    if args.output:
        folds.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
