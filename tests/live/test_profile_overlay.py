from pathlib import Path

import yaml


def test_live_canary_profile_overlay():
    """Test that live_canary profile overlay contains expected caps."""
    base = yaml.safe_load(Path("config/base.yaml").read_text())
    overlay = yaml.safe_load(Path("config/profiles/live_canary.yaml").read_text())

    # Merge overlay into base
    def deep_merge(a, b):
        out = dict(a)
        for k, v in (b or {}).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    cfg = deep_merge(base, overlay)

    # Assert expected caps are present and sane
    live = cfg.get("live", {})
    assert float(live.get("equity", 0)) > 0, "equity must be positive"
    assert 0 < float(live.get("per_trade_notional_pct", 0)) <= 0.1, (
        "per_trade_notional_pct must be 0-10%"
    )
    assert 0 < float(live.get("notional_daily_cap_pct", 0)) <= 0.5, (
        "notional_daily_cap_pct must be 0-50%"
    )

    # Assert models enabled
    models = cfg.get("models", {})
    assert models.get("enable") is True, "models must be enabled in live_canary"


def test_paper_strict_profile_overlay():
    """Test that paper_strict profile overlay contains expected caps."""
    base = yaml.safe_load(Path("config/base.yaml").read_text())
    overlay = yaml.safe_load(Path("config/profiles/paper_strict.yaml").read_text())

    # Merge overlay into base
    def deep_merge(a, b):
        out = dict(a)
        for k, v in (b or {}).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    cfg = deep_merge(base, overlay)

    # Assert expected caps are present and sane
    paper = cfg.get("paper", {})
    assert 0 < float(paper.get("weight_spike_cap", 0)) <= 0.5, "weight_spike_cap must be 0-50%"
    assert 0 < float(paper.get("turnover_cap", 0)) <= 2.0, "turnover_cap must be 0-200%"

    # Assert models enabled
    models = cfg.get("models", {})
    assert models.get("enable") is True, "models must be enabled in paper_strict"
