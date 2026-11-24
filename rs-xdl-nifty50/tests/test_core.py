"""Checks that features, regimes, and a short synthetic run stay consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsxdl.features import build_panel
from rsxdl.pipeline import RunConfig, run_from_frames
from rsxdl.regimes import _order_states, forward_filter


def _prices(n: int = 420, names: tuple[str, ...] = ("AAA.NS", "BBB.NS", "CCC.NS")):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-02", periods=n)
    index = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=dates, name="^NSEI")
    close = {}
    volume = {}
    for name in names:
        shock = rng.normal(0.0003, 0.012, n)
        close[name] = 50 * np.exp(np.cumsum(shock))
        volume[name] = rng.integers(100_000, 500_000, n)
    return pd.DataFrame(close, index=dates), pd.DataFrame(volume, index=dates).astype(float), index


def test_target_is_the_next_return():
    close, volume, index = _prices(n=80, names=("AAA.NS",))
    panel = build_panel(close, volume, index)
    row = panel.iloc[10]
    price = close["AAA.NS"]
    today = row["date"]
    nxt = price.index[price.index.get_loc(today) + 1]
    expected = price.loc[nxt] / price.loc[today] - 1
    assert abs(row["target"] - expected) < 1e-12


def test_states_are_ordered_by_rising_return():
    class Dummy:
        means_ = np.array([[0.4, 0.1], [-0.8, 0.2], [0.1, 0.0]])
        covars_ = np.stack([np.eye(2), np.eye(2), np.eye(2)])
        startprob_ = np.array([0.2, 0.5, 0.3])
        transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

    ordered = _order_states(Dummy())
    assert ordered.means_[0, 0] < ordered.means_[1, 0] < ordered.means_[2, 0]
    assert ordered.startprob_[0] == 0.5


def test_forward_filter_matches_a_persistent_state():
    class Dummy:
        means_ = np.array([[-1.0, 0.0], [1.0, 0.0]])
        covars_ = np.stack([np.eye(2) * 0.05, np.eye(2) * 0.05])
        startprob_ = np.array([0.5, 0.5])
        transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])

    observations = np.vstack([np.tile([-1.0, 0.0], (15, 1)), np.tile([1.0, 0.0], (15, 1))])
    states = forward_filter(Dummy(), observations)
    assert len(states) == 30
    assert set(states).issubset({0, 1})
    assert states[10] == 0
    assert states[-1] == 1


def test_synthetic_nifty_run_writes_metrics(tmp_path):
    close, volume, index = _prices()
    config = RunConfig(
        output_dir=str(tmp_path),
        n_estimators=25,
        hmm_iter=15,
        min_regime_rows=40,
        shap_rows=40,
        lstm=False,
        top_n=2,
        test_frac=0.2,
        val_frac=0.1,
    )
    metrics = run_from_frames(close, volume, index, config)
    assert "rmse" in metrics["regime_switching"]
    assert (tmp_path / "rc_shap.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    shap = pd.read_csv(tmp_path / "rc_shap.csv")
    assert {"slice", "feature", "mean_shap", "rank"}.issubset(shap.columns)
    assert shap["slice"].str.startswith("regime_").any()
