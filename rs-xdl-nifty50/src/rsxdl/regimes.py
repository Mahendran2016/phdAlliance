"""Three-state regime filter for the Nifty index.

States are ordered by the mean of the return feature after fitting:
0 bear, 1 neutral, 2 bull. The filter is the forward (filtered) posterior,
so the state on date t does not use index data after t. Parameters are fit
on the training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def _order_states(model):
    order = np.argsort(model.means_[:, 0])
    model.means_ = model.means_[order]
    model.covars_ = model.covars_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    return model


def forward_filter(model, observations: np.ndarray) -> np.ndarray:
    """Most likely filtered state at each time, using data up to that time."""
    means = np.asarray(model.means_, dtype=float)
    covars = np.asarray(model.covars_, dtype=float)
    log_start = np.log(np.clip(model.startprob_, 1e-16, 1.0))
    log_trans = np.log(np.clip(model.transmat_, 1e-16, 1.0))
    states = np.empty(len(observations), dtype=int)
    log_alpha = log_start
    for t, point in enumerate(observations):
        emit = np.array(
            [multivariate_normal.logpdf(point, means[k], covars[k]) for k in range(len(means))]
        )
        if t == 0:
            log_alpha = log_start + emit
        else:
            log_alpha = emit + logsumexp(log_alpha[:, None] + log_trans, axis=0)
        states[t] = int(np.argmax(log_alpha))
    return states


def _kmeans_regimes(train: np.ndarray, full: np.ndarray, n_states: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    model.fit(train)
    order = np.argsort(model.cluster_centers_[:, 0])
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = model.predict(full)
    return np.array([remap[int(label)] for label in labels], dtype=int)


def fit_regimes(
    index_features: pd.DataFrame,
    train_dates: set,
    n_states: int = 3,
    seed: int = 42,
    n_iter: int = 100,
) -> tuple[pd.Series, str]:
    values = index_features[["ret", "vol"]].to_numpy(dtype=float)
    train_index = pd.DatetimeIndex(pd.to_datetime(list(train_dates)))
    if train_index.tz is not None:
        train_index = train_index.tz_convert(None)
    train_index = train_index.normalize()
    train_mask = index_features.index.normalize().isin(train_index)
    if train_mask.sum() < n_states * 30:
        raise RuntimeError("Not enough training sessions to fit regimes")
    train = values[train_mask]
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    full_z = (values - center) / scale
    train_z = (train - center) / scale
    method = "hmm"
    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
            min_covar=1e-3,
        )
        model.fit(train_z)
        _order_states(model)
        states = forward_filter(model, full_z)
    except Exception as exc:
        states = _kmeans_regimes(train_z, full_z, n_states, seed)
        method = f"kmeans_fallback:{exc.__class__.__name__}"
    series = pd.Series(states, index=index_features.index, name="regime")
    return series, method
