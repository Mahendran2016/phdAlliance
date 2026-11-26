"""Point-in-time features for each Nifty constituent.

Every column on date t uses prices and volume up to and including the close
on t. ``target`` is the next-session simple return, which is the forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_21",
    "vol_21",
    "rsi_14",
    "macd",
    "volume_z",
    "idx_ret_1",
    "idx_vol_21",
)

MACRO_FEATURE_COLUMNS: tuple[str, ...] = (
    "usdinr_ret",
    "india_vix_z",
    "us_vix_z",
    "gold_ret",
    "brent_ret",
    "sp500_ret",
    "dxy_ret",
    "us10y_chg",
    "repo_rate",
    "cpi_yoy",
)


def feature_columns(use_macro: bool = False) -> list[str]:
    base = list(FEATURE_COLUMNS)
    return base + list(MACRO_FEATURE_COLUMNS) if use_macro else base


def attach_macro_features(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro columns on ``date`` (values known at that session close)."""
    if macro.empty:
        return panel
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index).tz_localize(None).normalize()
    lookup = macro.reset_index(names="date")
    lookup["date"] = pd.to_datetime(lookup["date"]).dt.normalize()
    merged = panel.merge(lookup, on="date", how="left", suffixes=("", "_macro"))
    for col in MACRO_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative))


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd - macd.ewm(span=signal, adjust=False).mean()


def _stock_frame(close: pd.Series, volume: pd.Series, ticker: str) -> pd.DataFrame:
    ret = close.pct_change()
    volume_mean = volume.rolling(21).mean()
    volume_std = volume.rolling(21).std()
    frame = pd.DataFrame(
        {
            "date": close.index,
            "ticker": ticker,
            "ret_1": ret,
            "ret_5": close.pct_change(5),
            "ret_21": close.pct_change(21),
            "vol_21": ret.rolling(21).std(),
            "rsi_14": rsi(close),
            "macd": macd_histogram(close),
            "volume_z": (volume - volume_mean) / volume_std.replace(0, np.nan),
        }
    )
    frame["target"] = frame["ret_1"].shift(-1)
    return frame


def build_panel(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    index_close: pd.Series,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    index_close = index_close.dropna().sort_index()
    index_ret = index_close.pct_change()
    index_vol = index_ret.rolling(21).std()
    frames: list[pd.DataFrame] = []
    for ticker in close.columns:
        prices = close[ticker].dropna()
        if len(prices) < 40:
            continue
        shares = volume[ticker].reindex(prices.index) if ticker in volume.columns else pd.Series(np.nan, index=prices.index)
        frame = _stock_frame(prices, shares.astype(float), ticker)
        frame["idx_ret_1"] = index_ret.reindex(prices.index).to_numpy()
        frame["idx_vol_21"] = index_vol.reindex(prices.index).to_numpy()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No constituent had enough history to build features")
    panel = pd.concat(frames, ignore_index=True)
    if macro is not None:
        panel = attach_macro_features(panel, macro)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    required = list(FEATURE_COLUMNS) + (list(MACRO_FEATURE_COLUMNS) if macro is not None else []) + ["target"]
    panel = panel.dropna(subset=required).copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def index_feature_frame(index_close: pd.Series) -> pd.DataFrame:
    ret = index_close.pct_change()
    frame = pd.DataFrame({"ret": ret, "vol": ret.rolling(21).std()})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame
