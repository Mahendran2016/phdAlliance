from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_features(
    prices: pd.Series,
    volume: pd.Series | None = None,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Technical features for regime detection and return forecasting."""
    log_ret = np.log(prices / prices.shift(1))
    out = pd.DataFrame(index=prices.index)
    out["log_return"] = log_ret
    out["volatility"] = log_ret.rolling(rolling_window).std()
    out["momentum"] = prices.pct_change(rolling_window)
    if volume is not None:
        out["volume_z"] = (volume - volume.rolling(rolling_window).mean()) / (
            volume.rolling(rolling_window).std() + 1e-9
        )
    out["forward_return"] = log_ret.shift(-1)
    return out.dropna()


def merge_macro_features(
    market: pd.DataFrame,
    macro: pd.DataFrame,
    on: str = "date",
) -> pd.DataFrame:
    """Left-join macro columns (CPI, VIX, USDINR, etc.) onto market features."""
    return market.reset_index(names=on).merge(macro, on=on, how="left").set_index(on)


def make_synthetic_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """Synthetic NSE-like daily panel for local smoke tests without external data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    regimes = np.zeros(n_days, dtype=int)
    for i in range(1, n_days):
        if regimes[i - 1] == 0:
            regimes[i] = 0 if rng.random() < 0.85 else 1
        elif regimes[i - 1] == 1:
            regimes[i] = 1 if rng.random() < 0.9 else (2 if rng.random() < 0.5 else 0)
        else:
            regimes[i] = 2 if rng.random() < 0.8 else 1

    drift = np.select([regimes == 0, regimes == 1, regimes == 2], [-0.0008, 0.0002, 0.0012])
    vol = np.select([regimes == 0, regimes == 1, regimes == 2], [0.025, 0.015, 0.012])
    rets = drift + vol * rng.standard_normal(n_days)
    price = 100 * np.exp(np.cumsum(rets))
    volume = rng.integers(1_000_000, 5_000_000, size=n_days)

    frame = pd.DataFrame(
        {
            "date": dates,
            "close": price,
            "volume": volume,
            "cpi_yoy": 4.0 + rng.normal(0, 0.1, n_days).cumsum() * 0.01,
            "india_vix": 14 + rng.normal(0, 1, n_days).cumsum() * 0.05,
            "usdinr": 74 + rng.normal(0, 0.05, n_days).cumsum(),
        }
    ).set_index("date")
    feats = build_market_features(frame["close"], frame["volume"])
    macro = frame[["cpi_yoy", "india_vix", "usdinr"]].loc[feats.index]
    return feats.join(macro)