"""
Risk V2 - Institutional-Grade Risk Management
============================================

Per-trade risk budgeting, ATR-based stops, and portfolio position caps.
Deterministic and flag-controlled.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RiskV2Config:
    per_trade_risk_bps: float
    atr_window: int
    atr_multiplier: float
    max_position_pct: float
    portfolio_max_gross_pct: float
    min_trade_notional: float


def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range for ATR computation."""
    prev_close = df["Close"].shift(1)
    a = df["High"] - df["Low"]
    b = (df["High"] - prev_close).abs()
    c = (df["Low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Average True Range using Wilder's smoothing."""
    tr = true_range(df)
    # Wilder's smoothing via EMA(alpha=1/window) for determinism
    alpha = 1.0 / float(window)
    return tr.ewm(alpha=alpha, adjust=False).mean()


def _shares_from_risk(equity: float, price: float, stop_dist: float, risk_dollars: float) -> int:
    """Calculate position size in shares based on risk budget."""
    if stop_dist <= 0 or price <= 0:
        return 0
    return int(np.floor(risk_dollars / stop_dist))


def apply(symbol: str,
          bar_df: pd.DataFrame,     # tz-aware OHLCV, recent→latest indexed ascending
          desired_weight: float,    # from model/signal, [-1, 1]
          equity: float,
          open_positions: dict,     # {symbol: {"entry_price": float, "stop": float, "weight": float}}
          portfolio_gross_weight: float,
          cfg: RiskV2Config) -> dict:
    """
    Apply risk v2 rules to a trading decision.
    
    Returns a dict with:
      weight, veto, reason, stop, atr, risk_dollars, capped_weight, action
    Deterministic: no RNG, no time access.
    """
    out = {
        "symbol": symbol, "atr": None, "stop": None, "risk_dollars": 0.0,
        "capped_weight": 0.0, "weight": 0.0, "veto": False, "reason": "", "action": "HOLD"
    }

    if bar_df.empty or len(bar_df) < cfg.atr_window + 1:
        out.update(veto=True, reason="insufficient_history")
        return out

    price = float(bar_df["Close"].iloc[-1])
    _atr = float(atr(bar_df, cfg.atr_window).iloc[-1])
    out["atr"] = _atr

    # Existing position: maintain/exit on stop
    pos = open_positions.get(symbol)
    if pos:
        out["stop"] = float(pos["stop"])
        if price <= out["stop"] and pos["weight"] > 0:
            out.update(weight=0.0, action="EXIT_STOP_LONG")
            return out
        if price >= out["stop"] and pos["weight"] < 0:
            out.update(weight=0.0, action="EXIT_STOP_SHORT")
            return out
        # otherwise pass through desired but capped
    else:
        # New position sizing by risk
        risk_dollars = equity * (cfg.per_trade_risk_bps / 1e4)
        stop_dist = cfg.atr_multiplier * _atr
        shares = _shares_from_risk(equity, price, stop_dist, risk_dollars)
        notional = shares * price
        if notional < cfg.min_trade_notional:
            out.update(veto=True, reason="below_min_notional")
            return out

        # Convert shares to weight target (sign from desired_weight)
        sign = np.sign(desired_weight) if desired_weight != 0 else 0
        target_weight_from_risk = sign * (notional / equity)
        # Propose a stop from side
        if sign > 0:
            out["stop"] = price - stop_dist
        elif sign < 0:
            out["stop"] = price + stop_dist

        out["risk_dollars"] = risk_dollars
        out["capped_weight"] = float(np.clip(target_weight_from_risk, -cfg.max_position_pct/100, cfg.max_position_pct/100))

    # Apply caps for existing or new
    proposed = out["capped_weight"] if out["capped_weight"] else desired_weight
    proposed = float(np.clip(proposed, -cfg.max_position_pct/100, cfg.max_position_pct/100))

    # Portfolio gross cap (simple veto if would exceed; could scale instead)
    projected_gross = portfolio_gross_weight - abs(open_positions.get(symbol, {}).get("weight", 0.0)) + abs(proposed)
    if projected_gross * 100.0 > cfg.portfolio_max_gross_pct:
        out.update(veto=True, reason="portfolio_gross_cap")
        return out

    out["weight"] = proposed
    out["action"] = "ENTER_OR_ADJUST" if proposed != 0 else "HOLD"
    return out
