"""Decision-level fit and replay for the OptimusKG V3.2 Hetionet results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

REGIMES = ("warm-start", "compound-disjoint", "target-disjoint")


@dataclass(frozen=True)
class Residual:
    mu: float = 0.0
    sd: float = 1.0
    w: float = 0.0
    bias: float = 0.0
    active: bool = False

    def __post_init__(self) -> None:
        if self.sd <= 0 or not np.isfinite([self.mu, self.sd, self.w, self.bias]).all():
            raise ValueError("Residual parameters must be finite, with a positive scale")
        if not isinstance(self.active, bool):
            raise ValueError("'active' must be a boolean")


def _inputs(base, score, covered):
    base = np.asarray(base, dtype=float)
    score = np.asarray(score, dtype=float)
    covered = np.asarray(covered)
    if base.ndim != 1 or len(base) == 0 or not (base.shape == score.shape == covered.shape):
        raise ValueError("Expected three nonempty, aligned vectors")
    if not np.isin(covered, [0, 1]).all():
        raise ValueError("Coverage must be boolean or 0/1")
    mask = covered.astype(bool)
    if not np.isfinite(base).all() or not np.isfinite(score[mask]).all():
        raise ValueError("Base logits and covered KGE scores must be finite")
    return base, score, mask


def fit_base(validation: pd.DataFrame, columns: list[str]):
    """Fit the C=0.5 standardized logistic combiner on validation rows.

    Pass z alone, z and the cold-start support logit eh, or z with the
    16 warm-start channels g0 through g15. Feature preparation is upstream.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not columns or len(set(columns)) != len(columns) or "label" in columns:
        raise ValueError("Supply unique predictor columns, excluding label")
    x = validation[columns].to_numpy(dtype=float)
    y = validation["label"].to_numpy()
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation needs finite features and both binary classes")
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(x, y)


def fit_hetionet_residual(base, label, score, covered) -> Residual:
    """Fit the fixed-base additive residual with the Hetionet validation recipe."""
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits

    base, score, covered = _inputs(base, score, covered)
    label = np.asarray(label)
    if label.shape != base.shape or not np.isin(label, [0, 1]).all():
        raise ValueError("Validation labels must be an aligned binary vector")
    y_cov = label[covered]
    if len(y_cov) < 200 or y_cov.sum() < 10 or (y_cov == 0).sum() < 10:
        return Residual()
    mu = float(score[covered].mean())
    sd = float(score[covered].std() + 1e-6)
    standardized = np.zeros_like(score)
    standardized[covered] = (score[covered] - mu) / sd
    b, y, s, mask = [
        torch.as_tensor(v, dtype=torch.float64) for v in (base, label, standardized, covered)
    ]
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = binary_cross_entropy_with_logits(b + mask * (params[0] * s + params[1]), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    w, bias = params.detach().tolist()
    return Residual(mu, sd, w, bias, True)


def predict(base, score, covered, residual: Residual, regime: str) -> dict[str, np.ndarray]:
    """Apply the reported Hetionet setting; target-disjoint uses the base decision."""
    if regime not in REGIMES:
        raise ValueError(f"Unknown Hetionet regime: {regime}")
    base, score, mask = _inputs(base, score, covered)
    delta = np.zeros_like(base)
    if residual.active and regime != "target-disjoint":
        delta[mask] = residual.w * (score[mask] - residual.mu) / residual.sd + residual.bias
    logit = base + delta
    if not np.isfinite(logit).all():
        raise ValueError("Residual produced a nonfinite logit")
    return {"base_logit": base, "residual": delta, "logit": logit, "probability": expit(logit)}


def aggregate_hetionet(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute Table 5 DEF-DTI statistics from one metric row per released fold."""
    expected = {(r, f) for r in REGIMES for f in range(10)}
    pairs = list(zip(frame["regime"], frame["fold"]))
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("Expected exactly folds 0 through 9 in each Hetionet regime")
    metrics = frame[["auprc", "auroc"]]
    if not np.isfinite(metrics.to_numpy()).all() or not ((0 <= metrics) & (metrics <= 1)).all().all():
        raise ValueError("AUPRC and AUROC must be finite and in [0, 1]")
    result = frame.groupby("regime")[["auprc", "auroc"]].agg(["mean", "std"]).reindex(REGIMES)
    result.columns = ["_".join(col) for col in result.columns]
    return result.reset_index()
