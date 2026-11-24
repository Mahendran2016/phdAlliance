# phdAlliance

Regime-Switching Explainable Deep Learning (RS-XDL) code for the Indian equity (NSE) PhD line of work.

## Quick start

1. Install Python 3.11+ and create a venv.
2. `pip install -r requirements.txt`
3. Smoke run on synthetic data: `python -m rsxdl.pipeline`
4. With your NSE CSV (`date`, `close`, optional `volume`):  
   `python -m rsxdl.pipeline --data path/to/prices.csv`

Pipeline: **features → HMM regimes → XGBoost forecast → SHAP → RC-SHAP → portfolio tilt**.

## Layout

- `rsxdl/regime/` — HMM regime detection  
- `rsxdl/models/` — forecast models (XGBoost first; LSTM/TFT next)  
- `rsxdl/explain/` — SHAP and Regime-Conditioned SHAP (RC-SHAP)  
- `rsxdl/eval/` — RMSE, Sharpe, Diebold–Mariano helpers  
- `0-Proposal/` — research proposal decks/PDF