"""Download and cache Nifty 50 prices from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsxdl.universe import NIFTY_INDEX, yahoo_tickers


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _download(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {tickers}")
    return raw


def _field(raw: pd.DataFrame, ticker: str, field: str) -> pd.Series | None:
    if isinstance(raw.columns, pd.MultiIndex):
        if (ticker, field) in raw.columns:
            series = raw[(ticker, field)]
        elif (field, ticker) in raw.columns:
            series = raw[(field, ticker)]
        else:
            return None
    else:
        if ticker != raw.columns.name and field not in raw.columns:
            return None
        series = raw[field] if field in raw.columns else None
    if series is None:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return series


def _normalize_index(series: pd.Series) -> pd.Series:
    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    out = series.copy()
    out.index = index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_market(
    start: str = "2015-01-01",
    end: str | None = None,
    symbols: list[str] | None = None,
    cache_dir: Path | str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return close prices, volumes, and the Nifty index close.

    Frames are aligned on trading dates. Tickers with no Yahoo history are
    dropped. Results are cached as parquet under ``cache_dir``.
    """
    tickers = yahoo_tickers(symbols)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_{(end or 'latest')}_{len(tickers)}"
    cache_path = cache_dir / f"market_{tag}.parquet"
    if cache_path.exists():
        stored = pd.read_parquet(cache_path)
        close = stored.xs("close", axis=1, level=0)
        volume = stored.xs("volume", axis=1, level=0)
        index_close = stored[("index", NIFTY_INDEX)].dropna()
        return close, volume, index_close

    raw = _download(tickers + [NIFTY_INDEX], start, end)
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for ticker in tickers:
        close = _field(raw, ticker, "Close")
        volume = _field(raw, ticker, "Volume")
        if close is None or volume is None:
            continue
        closes[ticker] = _normalize_index(close)
        volumes[ticker] = _normalize_index(volume)

    index_raw = _field(raw, NIFTY_INDEX, "Close")
    if index_raw is None:
        raise RuntimeError("Nifty index ^NSEI was not returned by Yahoo Finance")
    index_close = _normalize_index(index_raw).rename(NIFTY_INDEX)
    if len(closes) < 5:
        raise RuntimeError(f"Only {len(closes)} Nifty constituents downloaded")

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(close.index)
    packed = pd.concat(
        {"close": close, "volume": volume, "index": index_close.to_frame()},
        axis=1,
    )
    packed.to_parquet(cache_path)
    return close, volume, index_close
