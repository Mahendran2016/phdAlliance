from __future__ import annotations

import numpy as np
from scipy import stats


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0 or np.std(r) < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / np.std(r))


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """DM test on squared forecast errors (loss differential)."""
    d = errors_a**2 - errors_b**2
    d_mean = np.mean(d)
    n = len(d)
    gamma0 = np.var(d, ddof=1)
    gamma = gamma0
    for k in range(1, h):
        gamma += 2 * (1 - k / h) * np.cov(d[:-k], d[k:])[0, 1]
    dm_stat = d_mean / np.sqrt(gamma / n + 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)