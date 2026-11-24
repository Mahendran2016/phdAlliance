"""Nifty 50 symbols used by RS-XDL.

Symbols follow the Nifty 50 constituent list published on Wikipedia
(page checked November 2026, including additions through September 2026).
Yahoo Finance tickers use the NSE suffix. The index itself is ``^NSEI``.
Membership changes; pass ``--tickers`` to override this list.
"""

from __future__ import annotations

NIFTY_INDEX = "^NSEI"

# NSE symbols, without the Yahoo ``.NS`` suffix.
NIFTY50_SYMBOLS: tuple[str, ...] = (
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "BSE",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDIGO",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SHRIRAMFIN",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TMPV",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
)


def yahoo_tickers(symbols: tuple[str, ...] | list[str] | None = None) -> list[str]:
    chosen = NIFTY50_SYMBOLS if symbols is None else tuple(symbols)
    return [f"{symbol}.NS" for symbol in chosen]
