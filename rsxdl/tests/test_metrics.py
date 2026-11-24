import numpy as np

from rsxdl.eval.metrics import diebold_mariano, sharpe_ratio


def test_sharpe_ratio_positive_mean():
    r = np.array([0.01, 0.02, 0.015, 0.005])
    assert sharpe_ratio(r) > 0


def test_diebold_mariano_returns_finite_stats():
    a = np.random.default_rng(0).normal(0, 1, 200)
    b = np.random.default_rng(1).normal(0, 1.2, 200)
    stat, p = diebold_mariano(a, b)
    assert np.isfinite(stat)
    assert 0 <= p <= 1