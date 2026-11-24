"""Regime-conditioned SHAP for the tree models.

Tree SHAP values come from XGBoost ``pred_contribs``, which is exact SHAP for
a gradient-boosted tree. RC-SHAP is the mean signed contribution of each
feature inside a market regime, plus the mean absolute contribution used to
rank features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import DMatrix


def tree_shap(model, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    matrix = DMatrix(frame[feature_columns], feature_names=feature_columns)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)
    return np.asarray(contributions[:, :-1], dtype=float)


def regime_conditioned_shap(
    models: dict,
    test: pd.DataFrame,
    feature_columns: list[str],
    max_rows: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict] = []
    rng_seed = seed
    blocks: list[tuple[str, pd.DataFrame, object]] = [("pooled", test, models["pooled"])]
    for regime, model in models.items():
        if regime == "pooled":
            continue
        subset = test[test["regime"] == regime]
        blocks.append((f"regime_{int(regime)}", subset, model))

    for label, subset, model in blocks:
        if subset.empty:
            continue
        sample = subset
        if len(sample) > max_rows:
            sample = sample.sample(max_rows, random_state=rng_seed)
        values = tree_shap(model, sample, feature_columns)
        mean_signed = values.mean(axis=0)
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(-mean_abs)
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "slice": label,
                    "feature": feature_columns[feature_index],
                    "mean_shap": float(mean_signed[feature_index]),
                    "mean_abs_shap": float(mean_abs[feature_index]),
                    "rank": rank,
                    "n": int(len(sample)),
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["slice", "rank"]).reset_index(drop=True)
