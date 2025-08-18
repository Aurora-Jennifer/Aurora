from __future__ import annotations
from typing import Dict, Any


def check_live_risk(equity: float, positions: Dict[str, float], prices: Dict[str, float], risk_cfg: Dict[str, Any]) -> Dict[str, Any]:
    max_lev = float(risk_cfg.get("max_leverage", 1.0))
    max_gross = float(risk_cfg.get("max_gross_exposure", 1.0))
    max_pos = float(risk_cfg.get("max_position_pct", 1.0))
    # Compute exposures
    gross = 0.0
    max_abs = 0.0
    for sym, qty in positions.items():
        w = abs(qty * prices.get(sym, 0.0)) / max(1e-9, equity)
        gross += w
        max_abs = max(max_abs, w)
    if max_lev > 3.0:
        return {"status": "FAIL", "code": "LEVERAGE_LIMIT"}
    if gross > max_gross + 1e-9:
        return {"status": "FAIL", "code": "GROSS_EXPOSURE_LIMIT"}
    if max_abs > max_pos + 1e-9:
        return {"status": "FAIL", "code": "POSITION_LIMIT"}
    return {"status": "OK"}


