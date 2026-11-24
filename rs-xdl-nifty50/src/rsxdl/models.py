"""Regime-switching XGBoost and a regime-conditioned LSTM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def _regressor(n_estimators: int, seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=seed,
    )


def train_regime_models(
    train: pd.DataFrame,
    feature_columns: list[str],
    n_estimators: int = 300,
    min_rows: int = 250,
    seed: int = 42,
) -> dict:
    pooled = _regressor(n_estimators, seed)
    pooled.fit(train[feature_columns], train["target"])
    models: dict = {"pooled": pooled}
    for regime, subset in train.groupby("regime"):
        regime = int(regime)
        if len(subset) < min_rows:
            models[regime] = pooled
            continue
        model = _regressor(n_estimators, seed)
        model.fit(subset[feature_columns], subset["target"])
        models[regime] = model
    return models


def predict_regime(models: dict, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    pooled = models["pooled"]
    predictions = pooled.predict(frame[feature_columns])
    regimes = frame["regime"].to_numpy()
    for regime, model in models.items():
        if regime == "pooled" or model is pooled:
            continue
        mask = regimes == regime
        if mask.any():
            predictions[mask] = model.predict(frame.loc[mask, feature_columns])
    return np.asarray(predictions, dtype=float)


def _date_set(values) -> set[pd.Timestamp]:
    converted = pd.to_datetime(list(values))
    return set(pd.DatetimeIndex(converted).tz_localize(None).normalize())


def _stamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    return stamp.normalize()


def _windows(
    panel: pd.DataFrame,
    feature_columns: list[str],
    scaled: np.ndarray,
    lookback: int,
    keep_dates: set,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = _date_set(keep_dates)
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    offset = 0
    for _, group in panel.groupby("ticker", sort=False):
        length = len(group)
        block = scaled[offset : offset + length]
        regimes = group["regime"].to_numpy(dtype=int)
        one_hot = np.eye(n_states, dtype=np.float32)[np.clip(regimes, 0, n_states - 1)]
        features = np.concatenate([block, one_hot], axis=1)
        dates = group["date"].to_numpy()
        target = group["target"].to_numpy(dtype=np.float32)
        for end in range(lookback - 1, length):
            if _stamp(dates[end]) in keep:
                inputs.append(features[end - lookback + 1 : end + 1])
                targets.append(float(target[end]))
        offset += length
    if not inputs:
        return np.empty((0, lookback, len(feature_columns) + n_states), dtype=np.float32), np.empty(0)
    return np.stack(inputs).astype(np.float32), np.asarray(targets, dtype=np.float32)


def train_lstm(
    panel: pd.DataFrame,
    feature_columns: list[str],
    train_dates: set,
    val_dates: set,
    test_dates: set,
    lookback: int = 20,
    epochs: int = 8,
    hidden: int = 32,
    seed: int = 42,
    n_states: int = 3,
    max_train_windows: int = 30000,
) -> pd.DataFrame | None:
    """Train one LSTM on regime one-hot plus scaled features. Returns test predictions."""
    try:
        import torch
        from torch import nn
    except ImportError:
        return None

    torch.manual_seed(seed)
    ordered = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    train_mask = ordered["date"].isin(train_dates).to_numpy()
    center = ordered.loc[train_mask, feature_columns].mean().to_numpy(dtype=float)
    scale = ordered.loc[train_mask, feature_columns].std().to_numpy(dtype=float)
    scale[scale < 1e-8] = 1.0
    scaled = (ordered[feature_columns].to_numpy(dtype=float) - center) / scale

    x_train, y_train = _windows(ordered, feature_columns, scaled, lookback, train_dates, n_states)
    x_val, y_val = _windows(ordered, feature_columns, scaled, lookback, val_dates, n_states)
    x_test, y_test = _windows(ordered, feature_columns, scaled, lookback, test_dates, n_states)
    if len(x_train) < lookback or len(x_test) == 0:
        return None
    if len(x_train) > max_train_windows:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(x_train), size=max_train_windows, replace=False)
        x_train, y_train = x_train[pick], y_train[pick]

    device = torch.device("cpu")

    class ReturnLSTM(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(batch)
            return self.head(output[:, -1, :]).squeeze(-1)

    network = ReturnLSTM(x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def _batches(x_array, y_array, batch_size=256, shuffle=False):
        order = np.arange(len(x_array))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            choice = order[start : start + batch_size]
            yield (
                torch.tensor(x_array[choice], device=device),
                torch.tensor(y_array[choice], device=device),
            )

    best_state = None
    best_val = np.inf
    patience = 0
    for _ in range(epochs):
        network.train()
        for batch_x, batch_y in _batches(x_train, y_train, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if len(x_val) == 0:
            best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
            continue
        network.eval()
        with torch.no_grad():
            val_pred = network(torch.tensor(x_val, device=device))
            val_loss = loss_fn(val_pred, torch.tensor(y_val, device=device)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        pred = network(torch.tensor(x_test, device=device)).cpu().numpy()

    rows = []
    cursor = 0
    keep = _date_set(test_dates)
    for _, group in ordered.groupby("ticker", sort=False):
        dates = group["date"].to_numpy()
        tickers = group["ticker"].to_numpy()
        for end in range(lookback - 1, len(group)):
            if _stamp(dates[end]) in keep:
                rows.append(
                    {
                        "date": _stamp(dates[end]),
                        "ticker": str(tickers[end]),
                        "lstm_pred": float(pred[cursor]),
                    }
                )
                cursor += 1
    if cursor != len(pred):
        raise RuntimeError("LSTM prediction rows do not match the test windows")
    return pd.DataFrame(rows)
