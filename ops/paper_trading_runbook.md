# Paper Trading Runbook - CS_XGB_Residuals_VolFix v1.0.0

## Overview

**Duration:** 20 trading days (4 weeks)  
**Start Date:** TBD (Manual approval required)  
**Universe:** Top 300 US equities  
**Strategy:** Cross-sectional XGBoost with residualization and volatility fixes  
**Target Equity:** $10,000,000 (simulated)

## Configuration Freeze

**CRITICAL: These parameters must remain identical to validation:**

- Universe: Top 300 tickers (same list as validation)
- Horizon: 10 business days
- Embargo: 3 business days  
- Features: 45 features from protected whitelist
- Costs: 6.5 bps total (5 bps slippage + equivalent fees)
- Neutralization: Size + sector enabled

## Risk Limits (Hard Limits)

| Parameter | Limit | Action if Breached |
|-----------|-------|-------------------|
| Gross Exposure | 30% | Trim positions |
| Per-Symbol Position | 2% of equity | Trim to limit |
| Max Positions | 50 concurrent | Block new entries |
| Max Trades/Day | 60 | Pause trading |
| Daily Loss | -2% of equity | **KILL SWITCH** - Halt & flatten |
| ADV Participation | 2% per symbol | Block oversized orders |

## Live Sanity Gates (Auto-Halt Conditions)

### 🔴 Kill Switch Triggers (Immediate Halt + Flatten)
1. **Daily P&L ≤ -2%** of equity
2. **System error** preventing proper operation
3. **Manual kill switch** activation

### 🟡 Alert Triggers (Pause & Investigate)
1. **Action entropy < 0.75** for 10 consecutive bars
2. **Weekly |realized - expected| > 3σ** PnL divergence  
3. **Position concentration** > 2% for any symbol
4. **Order rejection rate** > 10% in any session

## Daily Monitoring Checklist

### Pre-Market (Before 9:30 AM ET)
- [ ] Verify data feed connectivity
- [ ] Check model inference pipeline
- [ ] Validate feature calculations
- [ ] Review overnight positions
- [ ] Confirm risk limits active

### Intraday (Every 30 minutes)
- [ ] Monitor action entropy (target: >0.75)
- [ ] Check position concentration
- [ ] Review order fill quality vs assumptions
- [ ] Validate P&L attribution

### End-of-Day (After 4:00 PM ET)
- [ ] Calculate daily IC vs realized returns
- [ ] Review decile spread performance  
- [ ] Analyze factor exposures (size/sector)
- [ ] Log any guard breaches or alerts
- [ ] Update drift report

## Telemetry Requirements

**Log every bar (real-time):**
```json
{
  "timestamp": "2024-09-08T15:30:00Z",
  "action_entropy": 0.82,
  "top_decile_symbols": ["AAPL", "MSFT", ...],
  "bottom_decile_symbols": ["TSLA", "META", ...],
  "gross_exposure_pct": 28.5,
  "num_positions": 47,
  "realized_slippage_bps": 4.2,
  "expected_slippage_bps": 5.0,
  "size_factor_exposure": -0.02,
  "sector_factor_exposure": 0.01
}
```

**End-of-day summary:**
```json
{
  "date": "2024-09-08",
  "daily_ic": 0.018,
  "decile_spread": 0.024,
  "daily_pnl_pct": 0.15,
  "expected_pnl_pct": 0.12,
  "turnover": 0.085,
  "fill_quality": "good",
  "guards_breached": [],
  "alerts_triggered": []
}
```

## Promotion Criteria to Live Trading

**Must meet ALL criteria for 4-week period:**

| Metric | Threshold | Current Status |
|--------|-----------|----------------|
| Paper IC | ≥ 0.015 | TBD |
| Paper Sharpe | ≥ 0.30 | TBD |
| Turnover | Within policy (≤2.0/mo) | TBD |
| Guard breaches | ≤ 1 per week, none repeated | TBD |
| Realized slippage | ≤ assumed + 25% (median) | TBD |
| Factor R² stability | Within 20% of backtest | TBD |

## Emergency Procedures

### Daily Loss Kill Switch Activated
1. **Immediate:** Halt all new orders
2. **Within 5 min:** Flatten all positions at market
3. **Within 10 min:** Alert operations team
4. **Within 30 min:** Generate incident report
5. **Before next day:** Root cause analysis

### System/Data Issues
1. **Immediate:** Activate manual override mode
2. **Assess:** Can positions be safely held overnight?
3. **If NO:** Flatten positions manually
4. **If YES:** Pause new trading, monitor closely
5. **Resolution:** Full system validation before restart

### Alert Investigation Process
1. **Entropy floor breach:** Check model scoring distribution
2. **P&L divergence:** Review execution quality and slippage
3. **Position concentration:** Verify position sizing logic
4. **Factor drift:** Validate neutralization calculations

## Data Quality Gates

**Real-time validation (every bar):**
- [ ] All 45 features computed successfully
- [ ] No NaN/infinite values in scores
- [ ] Cross-sectional dispersion > minimum threshold
- [ ] Feature distributions within expected ranges

**Daily validation:**
- [ ] Universe composition matches reference
- [ ] Sector mappings current and consistent  
- [ ] No forward-looking data leakage
- [ ] Factor loadings within expected ranges

## Communication Plan

### Daily Updates (EOD)
- Operations team: P&L, positions, alerts
- Risk team: Exposures, guard status, exceptions
- Technology team: System performance, data quality

### Weekly Reports
- Strategy performance vs expectations
- Factor attribution analysis
- Execution quality assessment  
- Lessons learned and adjustments

### Escalation Matrix
- **Level 1:** Daily losses 1-2% → Notify operations
- **Level 2:** Kill switch activation → Notify management
- **Level 3:** Repeated failures → Strategy review

## Success Metrics

### Performance Targets
- IC: 0.015-0.025 (consistent with backtest)
- Sharpe: 0.30+ (net of costs)
- Turnover: <2.0x/month
- Maximum drawdown: <5% over 4 weeks

### Operational Targets  
- Uptime: >99.5%
- Order fill rate: >95%
- Data quality: >99.9%
- Alert response time: <5 minutes

## Post-Paper Trading

### If ALL criteria met:
1. Final validation report
2. Risk committee approval
3. Live trading authorization
4. Gradual ramp to full allocation

### If ANY criteria failed:
1. Detailed failure analysis
2. Strategy modifications
3. Extended paper trading
4. Re-validation required

---

**Document Status:** Draft  
**Last Updated:** 2024-09-08  
**Approved By:** Pending manual review  
**Next Review:** Post paper-trading completion
