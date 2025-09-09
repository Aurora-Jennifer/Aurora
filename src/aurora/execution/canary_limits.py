from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CanaryConfig:
    equity: float
    per_trade_notional_pct: float  # e.g., 0.01
    notional_daily_cap_pct: float  # e.g., 0.10


@dataclass
class Decision:
    action: str  # "OK" | "HOLD"
    reason: str | None = None


def enforce_per_trade_notional(symbol: str, qty: float, px: float, cfg: CanaryConfig) -> Decision:
    notional = abs(qty * px)
    cap = cfg.per_trade_notional_pct * cfg.equity
    if notional > cap:
        return Decision("HOLD", f"per_trade_cap_exceeded:{symbol}:{notional:.2f}>{cap:.2f}")
    return Decision("OK")


def enforce_daily_notional(
    symbol: str,
    day_notional_used: float,
    add_notional: float,
    cfg: CanaryConfig,
) -> Decision:
    cap = cfg.notional_daily_cap_pct * cfg.equity
    if day_notional_used + abs(add_notional) > cap:
        return Decision("HOLD", f"daily_notional_cap_exceeded:{symbol}")
    return Decision("OK")


def check_caps(symbol: str, qty: float, px: float, day_used: float, cfg: CanaryConfig) -> Decision:
    d1 = enforce_per_trade_notional(symbol, qty, px, cfg)
    if d1.action != "OK":
        return d1
    return enforce_daily_notional(symbol, day_used, abs(qty * px), cfg)
