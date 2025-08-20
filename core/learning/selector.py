"""Contextual bandit strategy selector (stubbed Thompson/epsilon-greedy)."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class StrategyContext:
    regime: str
    vol_bin: int
    trend_strength: float
    liquidity: float
    spread_bps: float
    time_bucket: int
    corr_cluster: int


class BanditSelector:
    def __init__(self, state_path: str = "state/selector.pkl", epsilon: float = 0.1):
        self.state_path = state_path
        self.epsilon = float(epsilon)
        self.priors: dict[
            tuple[str, str], tuple[float, float]
        ] = {}  # key=(strategy, regime) -> (alpha, beta)
        self._load()

    def _load(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "rb") as f:
                    self.priors = pickle.load(f)  # nosec B301  # trusted local artifact; not user-supplied data
            except Exception:
                self.priors = {}

    def _save(self) -> None:
        with open(self.state_path, "wb") as f:
            pickle.dump(self.priors, f)

    def recommend(self, context: StrategyContext, candidates: list[str]) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(candidates)
        scores = []
        for s in candidates:
            key = (s, context.regime)
            alpha, beta = self.priors.get(key, (1.0, 1.0))
            samples = np.random.beta(alpha, beta)
            scores.append((samples, s))
        scores.sort(reverse=True)
        return scores[0][1]

    def update(self, context: StrategyContext, chosen: str, reward: float) -> None:
        key = (chosen, context.regime)
        alpha, beta = self.priors.get(key, (1.0, 1.0))
        if reward >= 0:
            alpha += float(reward)
        else:
            beta += float(-reward)
        self.priors[key] = (alpha, beta)
        self._save()
