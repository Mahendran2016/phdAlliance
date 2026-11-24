"""End-to-end RS-XDL run on a price panel."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsxdl.evaluate import diebold_mariano, regression_metrics, sharpe_ratio, top_n_portfolio
from rsxdl.explain import regime_conditioned_shap
from rsxdl.features import FEATURE_COLUMNS, build_panel, index_feature_frame
from rsxdl.models import predict_regime, train_lstm, train_regime_models
from rsxdl.regimes import fit_regimes


@dataclass
class RunConfig:
    start: str = "2015-01-01"
    end: str | None = None
    symbols: list[str] | None = None
    cache_dir: str = "data"
    output_dir: str = "artifacts"
    n_states: int = 3
    val_frac: float = 0.1
    test_frac: float = 0.2
    n_estimators: int = 300
    hmm_iter: int = 100
    min_regime_rows: int = 250
    shap_rows: int = 800
    top_n: int = 10
    cost_bps: float = 10.0
    lstm: bool = True
    lstm_epochs: int = 8
    lstm_lookback: int = 20
    seed: int = 42

    def quick(self) -> "RunConfig":
        self.start = "2023-01-01"
        self.symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "LT"]
        self.n_estimators = 80
        self.hmm_iter = 30
        self.min_regime_rows = 80
        self.shap_rows = 150
        self.lstm_epochs = 2
        self.top_n = 3
        return self


def chronological_split(dates, val_frac: float, test_frac: float):
    unique = pd.DatetimeIndex(pd.to_datetime(pd.unique(dates))).sort_values()
    n = len(unique)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train := n - n_test - n_val:
        if n_train < 30:
            n_test = max(1, n // 5)
            n_val = max(1, n // 10)
    train = set(unique[: -n_test - n_val])
    val = set(unique[-n_test - n_val : -n_test])
    test = set(unique[-n_test:])
    return train, val, test


def _plot_regimes(index_close: pd.Series, regimes: pd.Series, path: Path) -> None:
    aligned = pd.concat([index_close.rename("index"), regimes.rename("regime")], axis=1).dropna()
    if aligned.empty:
        return
    colors = {0: "#b45309", 1: "#64748b", 2: "#047857"}
    labels = {0: "Bear", 1: "Neutral", 2: "Bull"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for regime, group in aligned.groupby("regime"):
        ax.scatter(
            group.index,
            group["index"],
            s=8,
            c=colors.get(int(regime), "#334155"),
            label=labels.get(int(regime), str(regime)),
        )
    ax.set_title("Nifty index coloured by filtered regime")
    ax.set_ylabel("Index level")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_from_frames(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    index_close: pd.Series,
    config: RunConfig | None = None,
) -> dict:
    config = config or RunConfig()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel = build_panel(close, volume, index_close)
    train_dates, val_dates, test_dates = chronological_split(panel["date"], config.val_frac, config.test_frac)
    index_features = index_feature_frame(index_close)
    regimes, regime_method = fit_regimes(
        index_features,
        train_dates,
        n_states=config.n_states,
        seed=config.seed,
        n_iter=config.hmm_iter,
    )
    panel = panel.copy()
    panel["regime"] = panel["date"].map(regimes)
    panel = panel.dropna(subset=["regime"]).copy()
    panel["regime"] = panel["regime"].astype(int)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    train = panel[panel["date"].isin(train_dates)].copy()
    test = panel[panel["date"].isin(test_dates)].copy()
    if train.empty or test.empty:
        raise RuntimeError("The date split produced an empty train or test panel")

    features = list(FEATURE_COLUMNS)
    models = train_regime_models(
        train,
        features,
        n_estimators=config.n_estimators,
        min_rows=config.min_regime_rows,
        seed=config.seed,
    )
    regime_pred = predict_regime(models, test, features)
    pooled_pred = models["pooled"].predict(test[features])
    metrics = {
        "regime_switching": regression_metrics(test["target"], regime_pred),
        "pooled": regression_metrics(test["target"], pooled_pred),
        "diebold_mariano_regime_vs_pooled": diebold_mariano(test["target"], regime_pred, pooled_pred),
        "diebold_mariano_regime_vs_zero": diebold_mariano(test["target"], regime_pred, np.zeros(len(test))),
    }
    book = top_n_portfolio(test, regime_pred, top_n=config.top_n, cost_bps=config.cost_bps)
    metrics["portfolio"] = {
        "sharpe_net": sharpe_ratio(book["net_return"]),
        "sharpe_gross": sharpe_ratio(book["gross_return"]),
        "sharpe_equal_weight": sharpe_ratio(book["equal_weight_return"]),
        "mean_daily_net": float(book["net_return"].mean()),
        "mean_names": float(book["n_names"].mean()),
    }
    metrics["sample"] = {
        "rows": int(len(panel)),
        "tickers": int(panel["ticker"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(pd.Timestamp(max(train_dates)).date()),
        "test_start": str(pd.Timestamp(min(test_dates)).date()),
        "regime_method": regime_method,
        "regime_counts": {str(k): int(v) for k, v in panel["regime"].value_counts().sort_index().items()},
    }

    shap_table = regime_conditioned_shap(models, test, features, max_rows=config.shap_rows, seed=config.seed)
    shap_table.to_csv(output / "rc_shap.csv", index=False)
    book.to_csv(output / "portfolio.csv", index=False)
    regimes.rename("regime").to_csv(output / "regimes.csv", index_label="date")
    predictions = test[["date", "ticker", "regime", "target"]].copy()
    predictions["regime_pred"] = regime_pred
    predictions["pooled_pred"] = pooled_pred

    if config.lstm:
        lstm_pred = train_lstm(
            panel,
            features,
            train_dates,
            val_dates,
            test_dates,
            lookback=config.lstm_lookback,
            epochs=config.lstm_epochs,
            seed=config.seed,
            n_states=config.n_states,
        )
        if lstm_pred is not None and not lstm_pred.empty:
            predictions = predictions.merge(lstm_pred, on=["date", "ticker"], how="left")
            comparable = predictions.dropna(subset=["lstm_pred"])
            if not comparable.empty:
                metrics["lstm"] = regression_metrics(comparable["target"], comparable["lstm_pred"])
                metrics["diebold_mariano_lstm_vs_regime"] = diebold_mariano(
                    comparable["target"], comparable["lstm_pred"], comparable["regime_pred"]
                )
        else:
            metrics["lstm"] = None

    predictions.to_csv(output / "predictions.csv", index=False)
    _plot_regimes(index_close, regimes, output / "regimes.png")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run(config: RunConfig | None = None) -> dict:
    from rsxdl.data import load_market

    config = config or RunConfig()
    close, volume, index_close = load_market(
        start=config.start,
        end=config.end,
        symbols=config.symbols,
        cache_dir=config.cache_dir,
    )
    return run_from_frames(close, volume, index_close, config)


def config_to_dict(config: RunConfig) -> dict:
    return asdict(config)
