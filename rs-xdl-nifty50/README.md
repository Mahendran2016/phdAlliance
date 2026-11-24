# RS-XDL for the Nifty 50

Regime-Switching Explainable Deep Learning for Indian large-cap equities.

The daily Nifty index is split into bear, neutral, and bull states with a three-state Gaussian hidden Markov model. A separate gradient-boosted tree is fit inside each state, and tree SHAP is averaged inside each state (RC-SHAP) so feature rankings can change with the regime. A small LSTM takes the same features plus a regime one-hot and is scored on the same held-out sessions.

Prices come from Yahoo Finance. This is research code, not a trading system.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run

Full Nifty 50 history from 2015 (download, HMM, trees, SHAP, LSTM):

```bash
rsxdl --start 2015-01-01
```

A shorter check on eight large-cap names:

```bash
rsxdl --quick
```

Trees only, if PyTorch is not needed:

```bash
rsxdl --quick --no-lstm
```

Outputs land in `artifacts/`:

| File | Contents |
| --- | --- |
| `metrics.json` | RMSE, R², directional accuracy, Sharpe, Diebold-Mariano |
| `rc_shap.csv` | Mean SHAP by regime |
| `regimes.csv` | Filtered state of the Nifty index |
| `regimes.png` | Index level coloured by state |
| `predictions.csv` | Test-set forecasts |
| `portfolio.csv` | Top-forecast book versus equal weight |

The constituent list is in `src/rsxdl/universe.py` (Nifty 50 membership as published through late 2026). Override it with `--tickers RELIANCE,TCS,INFY`.

## Method

1. Features on date t use only information known at that close: 1-, 5-, and 21-day returns, 21-day volatility, RSI, MACD histogram, volume z-score, and the same-day Nifty return and volatility.
2. The label is the next session's simple return.
3. The HMM is fit on training-window Nifty return and volatility, then filtered forward so a state's label does not see the future.
4. States are ordered by mean return: 0 bear, 1 neutral, 2 bull.
5. The last 10% of dates validate the LSTM. The last 20% are the test. Trees are fit only on the training dates.
6. The reported book holds the names with the strongest positive forecasts, equal-weighted, and charges 10 bps of two-sided turnover.

## Tests

```bash
python -m pytest
```
