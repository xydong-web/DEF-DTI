import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scrdti import ModelConfig, SCRDTI
from scrdti.defdti import Residual, aggregate_hetionet, fit_base, fit_hetionet_residual, predict
from scrdti.defdti_cli import fit, score
from scrdti.defdti_pair import load_pair_predictor


class TestDEFDTI(unittest.TestCase):
    def test_pair_only_checkpoint_load_requires_complete_hca_path(self):
        config = ModelConfig(use_graph_context=False, use_kge_refinement=False)
        original = SCRDTI(config).eval()
        state = original.state_dict()
        loaded = load_pair_predictor(state)
        inputs = tuple(torch.randn(2, dim) for dim in (768, 768, 1280))
        with torch.inference_mode():
            torch.testing.assert_close(loaded(*inputs).pair_logit, original(*inputs).pair_logit)
        with self.assertRaises(RuntimeError):
            load_pair_predictor({k: v for k, v in state.items() if k != "hca.type_embeddings"})
        with self.assertRaisesRegex(ValueError, "Unexpected active parameter"):
            load_pair_predictor(SCRDTI(replace(config, use_graph_context=True)).state_dict())

    def test_masks_and_held_out_target_recipe(self):
        base = np.array([-2.0, 0.0, 2.0])
        kge = np.array([4.0, np.nan, -2.0])
        mask = np.array([1, 0, 1])
        params = Residual(mu=2, sd=2, w=3, bias=0.5, active=True)
        out = predict(base, kge, mask, params, "compound-disjoint")
        np.testing.assert_array_equal(out["residual"], [3.5, 0.0, -5.5])
        np.testing.assert_array_equal(out["logit"], [1.5, 0.0, -3.5])
        target = predict(base, kge, mask, params, "target-disjoint")
        np.testing.assert_array_equal(target["logit"], base)
        with self.assertRaises(ValueError):
            predict(base, kge, [1, 1, 1], params, "warm-start")

    def test_fit_residual_recovers_signal_and_coverage_guard(self):
        rng = np.random.default_rng(31)
        score = rng.normal(size=700)
        labels = (score > 0).astype(int)
        base = np.zeros(len(score))
        covered = np.arange(len(score)) % 2 == 0
        fit = fit_hetionet_residual(base, labels, score, covered)
        self.assertTrue(fit.active)
        self.assertGreater(fit.w, 0)
        np.testing.assert_allclose(fit.mu, score[covered].mean())
        self.assertFalse(fit_hetionet_residual(base, labels, score, np.zeros(len(score))).active)

    def test_base_refit_retains_pair_and_context_as_separate_columns(self):
        z = np.tile([-1.0, 0.0, 1.0], 20)
        context = np.repeat([-2.0, 2.0], 30)
        train = pd.DataFrame({"z": z, "eh": context, "label": (context > 0).astype(int)})
        model = fit_base(train, ["z", "eh"])
        score = model.decision_function(train[["z", "eh"]].to_numpy())
        self.assertLess(score[0], score[-1])

    def test_summary_rejects_incomplete_folds(self):
        frame = pd.DataFrame({
            "regime": ["warm-start"] * 3, "fold": [0, 1, 2],
            "auprc": [0.2, 0.4, 0.6], "auroc": [0.4, 0.6, 0.8],
        })
        with self.assertRaises(ValueError):
            aggregate_hetionet(frame)

    def test_fit_and_score_separate_labels_and_reject_pair_overlap(self):
        with TemporaryDirectory() as tmp:
            valid_path, test_path, params, output = (
                Path(tmp) / f for f in ("valid.csv", "test.csv", "residual.json", "output.csv")
            )
            valid = pd.DataFrame({
                "molecule_id": ["01", "02"],
                "protein_id": ["T", "T"],
                "base_logit": [0.0, 1.0],
                "kge_score": [0.3, 0.3],
                "covered": [True, False],
                "label": [0, 1],
            })
            valid.to_csv(valid_path, index=False)
            features = valid.drop(columns="label")
            features.to_csv(test_path, index=False)
            fit_args = SimpleNamespace(
                valid=valid_path, score_pairs=test_path, regime="target-disjoint", output=params
            )
            score_args = SimpleNamespace(inputs=test_path, params=params, output=output)
            with self.assertRaisesRegex(ValueError, "overlap"):
                fit(fit_args)
            with self.assertRaisesRegex(ValueError, "without labels"):
                score(SimpleNamespace(inputs=valid_path, params=params, output=output))
            features.assign(molecule_id=["03", "04"]).to_csv(test_path, index=False)
            fit(fit_args)
            score(score_args)
            scored = pd.read_csv(output, dtype={"molecule_id": str})
            self.assertEqual(scored.molecule_id.tolist(), ["03", "04"])
            np.testing.assert_array_equal(scored.logit, features.base_logit)

    def test_released_folds_reproduce_rounded_paper_row(self):
        metrics = Path(__file__).resolve().parents[1] / "results/hetionet_table5_folds.csv"
        result = aggregate_hetionet(pd.read_csv(metrics))
        np.testing.assert_array_equal(
            result.iloc[:, 1:].to_numpy().round(3),
            [[0.842, 0.006, 0.977, 0.001], [0.620, 0.036, 0.897, 0.023],
             [0.686, 0.020, 0.938, 0.006]],
        )
