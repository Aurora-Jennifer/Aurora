from core.execution.canary_limits import (
    CanaryConfig,
    enforce_per_trade_notional,
    enforce_daily_notional,
    check_caps,
)


def test_per_trade_cap_blocks():
    cfg = CanaryConfig(equity=100_000.0, per_trade_notional_pct=0.01, notional_daily_cap_pct=0.10)
    # 1.5% notional > 1% cap → HOLD
    d = enforce_per_trade_notional("SPY", qty=15.0, px=100.0, cfg=cfg)
    assert d.action == "HOLD"
    assert "per_trade_cap_exceeded" in (d.reason or "")


def test_daily_notional_cap_blocks():
    cfg = CanaryConfig(equity=100_000.0, per_trade_notional_pct=0.02, notional_daily_cap_pct=0.10)
    # already used 9% equity; adding 2% → exceeds 10% cap
    d = enforce_daily_notional("SPY", day_notional_used=9000.0, add_notional=2000.0, cfg=cfg)
    assert d.action == "HOLD"
    assert "daily_notional_cap_exceeded" in (d.reason or "")


def test_check_caps_ok_then_block():
    cfg = CanaryConfig(equity=50_000.0, per_trade_notional_pct=0.02, notional_daily_cap_pct=0.10)
    # OK: 0.8% per-trade, cumulative stays under 10%
    d_ok = check_caps("SPY", qty=4.0, px=100.0, day_used=300.0, cfg=cfg)
    assert d_ok.action == "OK"
    # Block on per-trade: 4% > 2%
    d_block = check_caps("SPY", qty=20.0, px=100.0, day_used=0.0, cfg=cfg)
    assert d_block.action == "HOLD"


