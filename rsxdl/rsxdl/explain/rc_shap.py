from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap


@dataclass
class RCShapResult:
    global_mean_abs: dict[str, float]
    by_regime_mean_abs: dict[int, dict[str, float]]


def compute_shap_values(
    model, x: np.ndarray, sample_size: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """TreeSHAP for XGBoost; subsample rows for speed."""
    n = min(sample_size, len(x))
    idx = np.linspace(0, len(x) - 1, n, dtype=int)
    x_sub = x[idx]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_sub)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values), idx


def regime_conditioned_shap(
    shap_values: np.ndarray,
    regimes: np.ndarray,
    feature_names: list[str],
    n_regimes: int,
) -> RCShapResult:
    global_mean = np.abs(shap_values).mean(axis=0)
    by_regime: dict[int, dict[str, float]] = {}
    for regime in range(n_regimes):
        mask = regimes == regime
        if not np.any(mask):
            continue
        regime_mean = np.abs(shap_values[mask]).mean(axis=0)
        by_regime[regime] = dict(zip(feature_names, regime_mean.tolist()))
    return RCShapResult(
        global_mean_abs=dict(zip(feature_names, global_mean.tolist())),
        by_regime_mean_abs=by_regime,
    )