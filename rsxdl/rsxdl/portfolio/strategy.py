from __future__ import annotations

import numpy as np


def explainability_tilted_weights(
    predicted_returns: np.ndarray,
    feature_importance: dict[str, float],
    stability_score: float,
    max_gross: float = 1.0,
) -> np.ndarray:
    """
    Long-only tilt from forecast sign; scale by explainability stability.

    ponytail: single-asset vector only; extend to cross-sectional NSE book later.
    """
    tilt = max(0.0, min(1.0, stability_score))
    core = np.sign(predicted_returns) * np.abs(predicted_returns)
    importance_boost = 1.0 + 0.1 * np.mean(list(feature_importance.values()))
    w = core * tilt * importance_boost
    gross = np.sum(np.abs(w))
    if gross > max_gross:
        w = w * (max_gross / gross)
    return w