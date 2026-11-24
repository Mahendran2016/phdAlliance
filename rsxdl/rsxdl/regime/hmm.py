from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """Hidden Markov Model for bull / bear / crisis-style regimes."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self._scaler.fit_transform(x)
        self._hmm.fit(x_scaled)
        return self._hmm.predict(x_scaled)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self._scaler.transform(x)
        return self._hmm.predict(x_scaled)

    def smooth_states(self, states: np.ndarray) -> np.ndarray:
        """ponytail: majority filter omitted; raw Viterbi states only."""
        return states