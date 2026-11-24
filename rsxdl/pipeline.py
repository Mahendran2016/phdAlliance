from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rsxdl.config import RSXDLConfig
from rsxdl.data.features import make_synthetic_panel
from rsxdl.data.load import load_nse_panel
from rsxdl.eval.metrics import diebold_mariano, sharpe_ratio
from rsxdl.explain.rc_shap import compute_shap_values, regime_conditioned_shap
from rsxdl.models.forecast import XGBoostForecaster, regression_metrics
from rsxdl.portfolio.strategy import explainability_tilted_weights
from rsxdl.regime.hmm import RegimeDetector


def _train_test_split_idx(n: int, train_ratio: float) -> int:
    return max(10, int(n * train_ratio))


def run_rsxdl(config: RSXDLConfig) -> dict:
    """End-to-end: features → HMM → forecast → SHAP → RC-SHAP → portfolio metrics."""
    if config.data_path:
        panel = load_nse_panel(config.data_path)
    else:
        panel = make_synthetic_panel()

    feature_cols = [c for c in panel.columns if c != config.target_col]
    x_all = panel[feature_cols].to_numpy(dtype=float)
    y_all = panel[config.target_col].to_numpy(dtype=float)

    split = _train_test_split_idx(len(panel), config.train_ratio)
    x_train, x_test = x_all[:split], x_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    regime_model = RegimeDetector(
        n_regimes=config.n_regimes, random_state=config.hmm_random_state
    )
    regimes_train = regime_model.fit(x_train)
    regimes_test = regime_model.predict(x_test)

    forecaster = XGBoostForecaster(random_state=config.hmm_random_state)
    forecaster.fit(x_train, y_train)
    y_pred = forecaster.predict(x_test)

    metrics = regression_metrics(y_test, y_pred)
    strategy_returns = np.sign(y_pred) * y_test
    metrics["sharpe_ratio"] = sharpe_ratio(strategy_returns)
    metrics["dm_vs_zero_baseline"], metrics["dm_p_vs_zero"] = diebold_mariano(
        y_test - y_pred, y_test
    )

    shap_values, shap_idx = compute_shap_values(
        forecaster.model, x_test, sample_size=config.shap_sample_size
    )
    regimes_for_shap = regimes_test[shap_idx]
    rc = regime_conditioned_shap(
        shap_values,
        regimes_for_shap,
        feature_cols,
        config.n_regimes,
    )

    stability = float(
        np.mean(
            [
                np.std(list(rc.by_regime_mean_abs[r].values()))
                for r in rc.by_regime_mean_abs
            ]
        )
    )
    weights = explainability_tilted_weights(y_pred, rc.global_mean_abs, 1.0 - stability)

    return {
        "n_train": split,
        "n_test": len(y_test),
        "metrics": metrics,
        "regime_counts_test": {
            int(k): int(v)
            for k, v in pd.Series(regimes_test).value_counts().to_dict().items()
        },
        "rc_shap": {
            "global_mean_abs": rc.global_mean_abs,
            "by_regime_mean_abs": {
                config.regime_names.get(k, str(k)): v
                for k, v in rc.by_regime_mean_abs.items()
            },
        },
        "portfolio_weights_head": weights[:5].tolist(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run RS-XDL pipeline")
    parser.add_argument("--data", type=Path, default=None, help="NSE price CSV path")
    parser.add_argument("--model", default="xgboost", choices=["xgboost"])
    parser.add_argument("--regimes", type=int, default=3)
    args = parser.parse_args()

    cfg = RSXDLConfig(data_path=args.data, model_name=args.model, n_regimes=args.regimes)
    out = run_rsxdl(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()