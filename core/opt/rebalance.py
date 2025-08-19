"""Real-time micro-rebalancing utilities."""

from __future__ import annotations


def should_rebalance(current_vol: float, target_vol: float, threshold_pct: float) -> bool:
    if target_vol <= 0.0:
        return False
    deviation = abs(current_vol - target_vol) / target_vol * 100.0
    return deviation >= threshold_pct


class RegimeHysteresis:
    def __init__(self, steps: int = 3):
        self.steps_required = max(1, int(steps))
        self.counter = 0
        self.last_regime = None

    def accept(self, regime: str) -> bool:
        if self.last_regime is None:
            self.last_regime = regime
            self.counter = 0
            return True
        if regime == self.last_regime:
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.steps_required:
            self.last_regime = regime
            self.counter = 0
            return True
        return False
