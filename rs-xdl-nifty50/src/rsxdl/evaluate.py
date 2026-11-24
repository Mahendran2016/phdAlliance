"""Forecast and portfolio metrics for the Nifty experiment."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    error = y - yhat
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(error**2) / total) if total > 0 else float("nan")
    directional = float(np.mean(np.sign(y) == np.sign(yhat)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "directional_accuracy": directional}


def sharpe_ratio(returns) -> float:
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 2 or np.std(series, ddof=1) == 0:
        return float("nan")
    return float(series.mean() / np.std(series, ddof=1) * math.sqrt(252))


def diebold_mariano(actual, forecast_a, forecast_b) -> dict[str, float]:
    """Squared-error Diebold-Mariano test of forecast_a against forecast_b.

    A negative statistic means forecast_a has the smaller squared error.
    """
    y = np.asarray(actual, dtype=float)
    loss_gap = (y - np.asarray(forecast_a, dtype=float)) ** 2 - (y - np.asarray(forecast_b, dtype=float)) ** 2
    n = len(loss_gap)
    if n < 5:
        return {"dm": float("nan"), "pvalue": float("nan")}
    centered = loss_gap - loss_gap.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    gamma1 = float(np.dot(centered[1:], centered[:-1]) / n)
    variance = (gamma0 + 2 * gamma1) / n
    if variance <= 0:
        return {"dm": float("nan"), "pvalue": float("nan")}
    statistic = float(loss_gap.mean() / math.sqrt(variance))
    pvalue = float(math.erfc(abs(statistic) / math.sqrt(2)))
    return {"dm": statistic, "pvalue": pvalue}


def top_n_portfolio(test: pd.DataFrame, predicted: np.ndarray, top_n: int = 10, cost_bps: float = 10.0) -> pd.DataFrame:
    """Equal-weight the names with the highest positive next-day forecasts.

    The return earned on date t is the next session's return (``target``),
    which is known only after the close that produced the forecast.
    """
    frame = test[["date", "ticker", "target"]].copy()
    frame["pred"] = predicted
    records = []
    held: set[str] = set()
    for date, group in frame.groupby("date", sort=True):
        chosen = group.nlargest(min(top_n, len(group)), "pred")
        chosen = chosen[chosen["pred"] > 0]
        names = set(chosen["ticker"].astype(str))
        gross = float(chosen["target"].mean()) if len(chosen) else 0.0
        if not held and not names:
            turnover = 0.0
        else:
            overlap = len(held & names)
            turnover = 1.0 - overlap / max(len(held | names), 1)
        cost = turnover * (cost_bps / 10000.0)
        records.append(
            {
                "date": date,
                "gross_return": gross,
                "net_return": gross - cost,
                "turnover": turnover,
                "n_names": int(len(names)),
            }
        )
        held = names
    book = pd.DataFrame(records)
    benchmark = frame.groupby("date", sort=True)["target"].mean()
    book["equal_weight_return"] = book["date"].map(benchmark).astype(float)
    return book
