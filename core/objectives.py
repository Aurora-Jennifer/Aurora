"""
Objective functions for growth maximization under risk constraints.

Provides a pluggable API to compute an objective score from returns/equity and
derive a target risk budget to guide position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Objective(Protocol):
    def score(self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict) -> float: ...

    def derive_risk_budget(
        self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict
    ) -> tuple[float, float]: ...


@dataclass
class ObjectiveParams:
    max_leverage: float = 1.0
    max_gross_exposure_pct: float = 50.0
    kelly_cap_fraction: float = 0.25
    risk_aversion_lambda: float = 3.0
    downside_lambda: float = 2.0


class ExpectedLogUtility:
    """
    Kelly-style log utility with a cap via kelly_cap_fraction.
    Returns a score approximating expected log growth and a risk budget.
    """

    def __init__(self, params: ObjectiveParams):
        self.params = params

    def score(self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict) -> float:
        if daily_returns is None or len(daily_returns) == 0:
            return 0.0
        # Use small epsilon to avoid log(0)
        eps = 1e-6
        log_growth = np.log1p(np.clip(daily_returns.values, -0.99, 10.0))
        return float(np.nanmean(log_growth))

    def derive_risk_budget(
        self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict
    ) -> tuple[float, float]:
        # Estimate mean and variance
        mu = (
            float(np.nanmean(daily_returns))
            if daily_returns is not None and len(daily_returns) > 0
            else 0.0
        )
        sigma2 = (
            float(np.nanvar(daily_returns))
            if daily_returns is not None and len(daily_returns) > 1
            else 0.0
        )
        # Kelly fraction ~ mu / sigma^2 (bounded and scaled by cap)
        kelly = 0.0 if sigma2 <= 1e-12 else np.clip(mu / sigma2, 0.0, 10.0)
        fractional_kelly = kelly * self.params.kelly_cap_fraction
        # Risk budget as target annualized volatility fraction of cap (0..1)
        # Map fractional_kelly (open range) to [0, 1.5] then clip
        risk_budget = float(np.clip(fractional_kelly, 0.0, 1.5))
        # position multiplier to scale nominal sizes
        pos_mult = float(np.clip(1.0 + 0.5 * (fractional_kelly - 0.5), 0.5, 1.5))
        return risk_budget, pos_mult


class MeanVariance:
    """Maximize mu - lambda * sigma^2."""

    def __init__(self, params: ObjectiveParams):
        self.params = params

    def score(self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict) -> float:
        if daily_returns is None or len(daily_returns) == 0:
            return 0.0
        mu = float(np.nanmean(daily_returns))
        sigma2 = float(np.nanvar(daily_returns))
        return mu - self.params.risk_aversion_lambda * sigma2

    def derive_risk_budget(
        self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict
    ) -> tuple[float, float]:
        mu = (
            float(np.nanmean(daily_returns))
            if daily_returns is not None and len(daily_returns) > 0
            else 0.0
        )
        sigma = (
            float(np.sqrt(np.nanvar(daily_returns)))
            if daily_returns is not None and len(daily_returns) > 1
            else 0.0
        )
        # Translate mean-variance tradeoff into a vol target scalar in [0, 1.5]
        desirability = mu / (1e-6 + self.params.risk_aversion_lambda * (sigma**2))
        risk_budget = float(np.clip(desirability, 0.0, 1.5))
        pos_mult = float(np.clip(1.0 + 0.5 * (desirability - 0.5), 0.5, 1.5))
        return risk_budget, pos_mult


class SortinoUtility:
    """Maximize Sortino with downside penalty."""

    def __init__(self, params: ObjectiveParams):
        self.params = params

    def score(self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict) -> float:
        if daily_returns is None or len(daily_returns) == 0:
            return 0.0
        r = daily_returns.values
        downside = r[r < 0.0]
        downside_dev = float(np.sqrt(np.nanmean(np.square(downside)))) if downside.size > 0 else 0.0
        mu = float(np.nanmean(r))
        sortino = 0.0 if downside_dev == 0.0 else mu / downside_dev
        return sortino - self.params.downside_lambda * downside_dev

    def derive_risk_budget(
        self, daily_returns: pd.Series, equity: pd.Series, risk_metrics: dict
    ) -> tuple[float, float]:
        r = daily_returns.values if daily_returns is not None else np.array([])
        mu = float(np.nanmean(r)) if r.size > 0 else 0.0
        downside = r[r < 0.0]
        downside_dev = float(np.sqrt(np.nanmean(np.square(downside)))) if downside.size > 0 else 0.0
        # Favor higher budgets when upside mean dominates downside risk
        signal = mu - self.params.downside_lambda * downside_dev
        risk_budget = float(np.clip(0.5 + signal, 0.0, 1.5))
        pos_mult = float(np.clip(1.0 + 0.5 * signal, 0.5, 1.5))
        return risk_budget, pos_mult


def build_objective(config: dict) -> Objective:
    obj_cfg = config.get("objective", {})
    params = ObjectiveParams(
        max_leverage=float(config.get("risk_params", {}).get("max_leverage", 1.0)),
        max_gross_exposure_pct=float(
            config.get("risk_params", {}).get("max_gross_exposure_pct", 50.0)
        ),
        kelly_cap_fraction=float(obj_cfg.get("kelly_cap_fraction", 0.25)),
        risk_aversion_lambda=float(obj_cfg.get("risk_aversion_lambda", 3.0)),
        downside_lambda=float(obj_cfg.get("downside_lambda", 2.0)),
    )
    obj_type = obj_cfg.get("type", "log_utility").lower()
    if obj_type in ("log_utility", "log", "kelly"):
        return ExpectedLogUtility(params)
    if obj_type in ("mean_variance", "mv"):
        return MeanVariance(params)
    if obj_type in ("sortino", "sortino_utility"):
        return SortinoUtility(params)
    # default
    return ExpectedLogUtility(params)
