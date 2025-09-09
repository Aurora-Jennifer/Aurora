import yaml


def _read_costs_yaml() -> dict:
    try:
        with open("config/components/costs.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_cost_profile(profile: str | None = None) -> dict:
    """Load a cost profile from costs.yaml.

    Falls back to top-level keys if the profile is not found.
    """
    cfg = _read_costs_yaml()
    if profile:
        prof = (cfg.get("profiles") or {}).get(profile)
        if prof:
            return prof
    # Fallback to top-level keys
    return {
        "commission_bps": float(cfg.get("commission_bps", 0.0)),
        "half_spread_bps": float(cfg.get("half_spread_bps", cfg.get("spread_bps", 0.0))),
        "slippage_bps_per_turnover": float(
            cfg.get("slippage_bps_per_turnover", cfg.get("slippage_bps", 0.0))
        ),
        "borrow_bps": float(cfg.get("borrow_bps", 0.0)),
    }


def total_bps(prof: dict) -> float:
    """Sum static bps (commission + half-spread + slippage if provided as constant + borrow).
    For per-turnover slippage, prefer cost_penalty_from_turnover.
    """
    return float(
        prof.get("commission_bps", 0.0)
        + prof.get("half_spread_bps", prof.get("spread_bps", 0.0))
        + prof.get("slippage_bps", 0.0)
        + prof.get("borrow_bps", 0.0)
    )


def cost_penalty_from_turnover(prof: dict, turnover: float) -> float:
    """Compute fractional penalty to deduct from IC using turnover-scaled costs.

    Returns a fraction (bps/10_000) that can be subtracted from IC.
    """
    commission = float(prof.get("commission_bps", 0.0))
    half_spread = float(prof.get("half_spread_bps", prof.get("spread_bps", 0.0)))
    slip_per_turn = float(prof.get("slippage_bps_per_turnover", prof.get("slippage_bps", 0.0)))
    borrow = float(prof.get("borrow_bps", 0.0))
    total_bps = commission + half_spread + borrow + slip_per_turn * float(turnover)
    return total_bps / 10_000.0


