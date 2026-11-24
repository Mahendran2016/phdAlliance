from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RSXDLConfig:
    """Runtime settings for the RS-XDL pipeline."""

    n_regimes: int = 3
    hmm_random_state: int = 42
    forecast_horizon: int = 1
    train_ratio: float = 0.8
    rolling_window: int = 20
    model_name: str = "xgboost"
    shap_sample_size: int = 200
    data_path: Path | None = None
    target_col: str = "forward_return"
    regime_names: dict[int, str] = field(
        default_factory=lambda: {0: "bear", 1: "neutral", 2: "bull"}
    )