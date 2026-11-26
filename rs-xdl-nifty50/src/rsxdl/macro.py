"""Macro and fear gauges aligned to the Nifty calendar (point-in-time)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MACRO_TICKERS = {
    "USDINR": "INR=X",
    "GOLD": "GC=F",
    "BRENT": "BZ=F",
    "SP500": "^GSPC",
    "US_VIX": "^VIX",
    "INDIA_VIX": "^INDIAVIX",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
}

RBI_REPO_SCHEDULE = [
    ("2015-01-01", 7.75),
    ("2015-01-15", 7.75),
    ("2015-03-04", 7.50),
    ("2015-06-02", 7.25),
    ("2016-04-05", 6.50),
    ("2017-08-02", 6.00),
    ("2018-06-06", 6.25),
    ("2018-08-01", 6.50),
    ("2019-02-07", 6.25),
    ("2019-08-07", 5.40),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2026-12-31", 6.50),
]

CPI_DATA = [
    ("2015-01-12", 5.11),
    ("2016-01-12", 5.61),
    ("2017-01-12", 3.41),
    ("2018-01-12", 5.21),
    ("2019-01-14", 2.05),
    ("2020-01-13", 7.59),
    ("2021-01-12", 4.06),
    ("2022-01-12", 6.01),
    ("2023-01-12", 6.52),
    ("2024-01-12", 5.10),
    ("2024-12-13", 5.48),
]


def _normalize_index(series: pd.Series) -> pd.Series:
    idx = pd.to_datetime(series.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out = series.copy()
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def build_repo_rate_series(date_index: pd.DatetimeIndex) -> pd.Series:
    repo = pd.Series(index=date_index, dtype=float, name="repo_rate")
    for i, (date_str, rate) in enumerate(RBI_REPO_SCHEDULE):
        start_d = pd.Timestamp(date_str)
        end_d = (
            pd.Timestamp(RBI_REPO_SCHEDULE[i + 1][0])
            if i + 1 < len(RBI_REPO_SCHEDULE)
            else date_index[-1] + pd.Timedelta(days=1)
        )
        mask = (date_index >= start_d) & (date_index < end_d)
        repo.loc[mask] = rate
    return repo.ffill()


def build_cpi_series(date_index: pd.DatetimeIndex) -> pd.Series:
    """CPI YoY is visible only from MOSPI publication dates onward."""
    cpi = pd.Series(index=date_index, dtype=float, name="cpi_yoy")
    for pub_date_str, value in CPI_DATA:
        cpi.loc[date_index >= pd.Timestamp(pub_date_str)] = value
    return cpi


def _download_closes(tickers: dict[str, str], start: str, end: str | None) -> dict[str, pd.Series]:
    import yfinance as yf

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Macro download returned no rows")
    out: dict[str, pd.Series] = {}
    for name, ticker in tickers.items():
        if isinstance(raw.columns, pd.MultiIndex):
            if (ticker, "Close") in raw.columns:
                series = raw[(ticker, "Close")]
            elif ("Close", ticker) in raw.columns:
                series = raw[("Close", ticker)]
            else:
                continue
        else:
            series = raw["Close"]
        series = _normalize_index(pd.to_numeric(series, errors="coerce").dropna())
        if not series.empty:
            out[name] = series.rename(name)
    return out


def _macro_feature_frame(levels: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=levels.index)
    for col in ("USDINR", "GOLD", "BRENT", "SP500", "DXY"):
        if col in levels.columns:
            frame[f"{col.lower()}_ret"] = np.log(levels[col] / levels[col].shift(1))
    for col in ("INDIA_VIX", "US_VIX"):
        if col in levels.columns:
            z = (levels[col] - levels[col].rolling(21).mean()) / levels[col].rolling(21).std()
            frame[f"{col.lower()}_z"] = z
    if "US10Y" in levels.columns:
        frame["us10y_chg"] = levels["US10Y"].diff()
    frame["repo_rate"] = build_repo_rate_series(frame.index)
    frame["cpi_yoy"] = build_cpi_series(frame.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def load_macro_panel(
    calendar: pd.DatetimeIndex,
    start: str,
    end: str | None = None,
    cache_dir: Path | str = "data",
) -> pd.DataFrame:
    """Daily macro features indexed by session date, aligned to ``calendar``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"macro_{start}_{end or 'latest'}"
    cache_path = cache_dir / f"{tag}.parquet"
    if cache_path.exists():
        stored = pd.read_parquet(cache_path)
        return stored.reindex(pd.to_datetime(calendar).normalize())

    levels = pd.DataFrame(_download_closes(MACRO_TICKERS, start, end))
    levels = levels.reindex(pd.to_datetime(calendar).normalize()).ffill()
    features = _macro_feature_frame(levels)
    features.to_parquet(cache_path)
    return features