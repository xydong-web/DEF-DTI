"""Pair-only HCA predictor for frozen DEF-DTI embeddings."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from defdti.compat.legacy import _map_key
from defdti.config import ModelConfig
from defdti.model import DEFDTI


def load_pair_predictor(state: Mapping[str, torch.Tensor]) -> DEFDTI:
    """Load the active pair path from a release or original pair-only state dict."""
    model = DEFDTI(ModelConfig(use_graph_context=False, use_kge_refinement=False))
    expected = model.state_dict().keys()
    active_modules = {key.split(".", 1)[0] for key in expected}
    active_modules.update(("graph_context", "kge_refinement"))
    inactive_buffers = {"kge_refinement.norm_mean", "kge_refinement.norm_std"}
    mapped = {}
    for name, value in state.items():
        key = name if name in expected else _map_key(name)
        if key in expected:
            if key in mapped:
                raise ValueError(f"Duplicate HCA parameter: {key}")
            mapped[key] = value
        elif (key is not None and key not in inactive_buffers) or (
            name.split(".", 1)[0] in active_modules
        ):
            raise ValueError(f"Unexpected active parameter in pair-only checkpoint: {name}")
    model.load_state_dict(mapped, strict=True)
    return model.eval()
