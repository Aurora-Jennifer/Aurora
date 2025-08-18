from __future__ import annotations
from typing import Dict, Tuple


def clamp_weights(w: Dict[str, float], max_pos: float, max_gross: float) -> Tuple[Dict[str, float], dict]:
    """Clamp per-position and gross exposure. Returns (clamped, stats)."""
    gross_before = sum(abs(x) for x in w.values())
    if max_pos <= 0 or max_gross <= 0:
        return dict(w), {"clamped": False, "gross_before": gross_before, "gross_after": gross_before}

    # Per-position cap
    capped = {k: max(-max_pos, min(max_pos, v)) for k, v in w.items()}
    gross = sum(abs(v) for v in capped.values())
    scale = 1.0 if gross <= max_gross else (max_gross / gross if gross > 0 else 1.0)
    clamped = {k: v * scale for k, v in capped.items()}
    stats = {
        "clamped": scale < 0.999 or capped != w,
        "per_pos_cap": max_pos,
        "gross_cap": max_gross,
        "gross_before": gross_before,
        "gross_after": sum(abs(v) for v in clamped.values()),
        "scale": scale,
    }
    return clamped, stats


