from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsxdl.data.features import build_market_features, merge_macro_features


def load_nse_panel(
    price_csv: Path,
    macro_csv: Path | None = None,
    date_col: str = "date",
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Load NSE equity panel from CSV.

    price_csv columns: date, close, volume (optional).
    macro_csv columns: date, cpi_yoy, india_vix, usdinr (any subset).
    """
    prices = pd.read_csv(price_csv, parse_dates=[date_col]).sort_values(date_col)
    feats = build_market_features(
        prices.set_index(date_col)[price_col],
        prices.set_index(date_col)[volume_col] if volume_col in prices.columns else None,
    )
    if macro_csv is None:
        return feats
    macro = pd.read_csv(macro_csv, parse_dates=[date_col]).sort_values(date_col)
    return merge_macro_features(feats, macro, on=date_col)