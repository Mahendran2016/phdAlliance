"""Expanding-window walk-forward evaluation with per-fold regime refit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsxdl.evaluate import diebold_mariano, regression_metrics
from rsxdl.features import feature_columns, index_feature_frame
from rsxdl.models import predict_regime, train_regime_models
from rsxdl.regimes import fit_regimes


def walk_forward_date_folds(
    dates: pd.Series,
    n_folds: int = 5,
    min_train_frac: float = 0.6,
) -> list[tuple[set, set]]:
    unique = pd.DatetimeIndex(pd.to_datetime(pd.unique(dates))).sort_values()
    n = len(unique)
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    init = max(30, int(n * min_train_frac))
    span = n - init
    if span < n_folds:
        n_folds = max(1, span)
    fold_size = max(1, span // n_folds)
    folds: list[tuple[set, set]] = []
    for k in range(n_folds):
        tr_end = init + k * fold_size
        te_end = min(tr_end + fold_size, n)
        if te_end <= tr_end:
            break
        train = set(unique[:tr_end])
        test = set(unique[tr_end:te_end])
        if train and test:
            folds.append((train, test))
    return folds


def _eval_one_fold(
    panel: pd.DataFrame,
    index_close: pd.Series,
    train_dates: set,
    test_dates: set,
    features: list[str],
    config,
) -> dict:
    index_features = index_feature_frame(index_close)
    regimes, method = fit_regimes(
        index_features,
        train_dates,
        n_states=config.n_states,
        seed=config.seed,
        n_iter=config.hmm_iter,
    )
    frame = panel.copy()
    frame["regime"] = frame["date"].map(regimes)
    frame = frame.dropna(subset=["regime"]).copy()
    frame["regime"] = frame["regime"].astype(int)
    train = frame[frame["date"].isin(train_dates)]
    test = frame[frame["date"].isin(test_dates)]
    if train.empty or test.empty:
        raise RuntimeError("Walk-forward fold produced an empty train or test slice")
    models = train_regime_models(
        train,
        features,
        n_estimators=config.n_estimators,
        min_rows=config.min_regime_rows,
        seed=config.seed,
    )
    regime_pred = predict_regime(models, test, features)
    pooled_pred = models["pooled"].predict(test[features])
    return {
        "regime_switching": regression_metrics(test["target"], regime_pred),
        "pooled": regression_metrics(test["target"], pooled_pred),
        "diebold_mariano_regime_vs_pooled": diebold_mariano(test["target"], regime_pred, pooled_pred),
        "regime_method": method,
        "train_days": int(len(train_dates)),
        "test_days": int(len(test_dates)),
        "test_rows": int(len(test)),
    }


def run_walk_forward(
    panel: pd.DataFrame,
    index_close: pd.Series,
    config,
    output: Path | None = None,
) -> dict:
    features = feature_columns(config.use_macro)
    folds = walk_forward_date_folds(
        panel["date"],
        n_folds=config.walk_forward_folds,
        min_train_frac=config.walk_forward_min_train_frac,
    )
    if not folds:
        raise RuntimeError("Walk-forward produced no folds; lengthen history or reduce min_train_frac")

    rows: list[dict] = []
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        fold_metrics = _eval_one_fold(panel, index_close, train_dates, test_dates, features, config)
        row = {"fold": fold_id, **fold_metrics["regime_switching"]}
        row["regime_method"] = fold_metrics["regime_method"]
        row["train_days"] = fold_metrics["train_days"]
        row["test_days"] = fold_metrics["test_days"]
        row["test_rows"] = fold_metrics["test_rows"]
        dm = fold_metrics["diebold_mariano_regime_vs_pooled"]
        row["dm_stat"] = dm.get("dm")
        row["dm_pvalue"] = dm.get("pvalue")
        rows.append(row)

    table = pd.DataFrame(rows)
    aggregate = {
        "rmse_mean": float(table["rmse"].mean()),
        "rmse_std": float(table["rmse"].std(ddof=0)),
        "r2_mean": float(table["r2"].mean()),
        "directional_accuracy_mean": float(table["directional_accuracy"].mean()),
        "n_folds": int(len(table)),
    }

    if output is not None:
        table.to_csv(output / "walk_forward_folds.csv", index=False)
        import json

        summary = {"aggregate": aggregate, "folds": rows}
        (output / "walk_forward_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"aggregate": aggregate, "folds": rows}