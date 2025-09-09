# Paper Trading Setup Guide

## 🚨 CRITICAL: REAL DATA INTEGRATION REQUIRED

Your system is currently running on **mock data**. For actual paper trading validation, you need real market data integration.

## 📋 IMMEDIATE NEXT STEPS (Before 20-Day Validation)

### 1. 🔌 INTEGRATE REAL DATA SOURCE

**Choose your data provider:**
- **yfinance** (free, good for testing)
- **Alpha Vantage** (free tier available)
- **Polygon.io** (professional, paid)
- **Interactive Brokers** (if using IB for trading)

**Implementation required:**
```python
# Create: ml/data_provider.py
class RealDataProvider:
    def fetch_daily_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        # Replace mock data with real API calls
        pass
    
    def get_latest_prices(self, symbols: List[str]) -> pd.DataFrame:
        # Real-time or delayed quotes
        pass
```

### 2. 🏗️ UPDATE PIPELINE FOR REAL DATA

**Files to modify:**
- `ml/panel_builder.py` - Replace mock data creation
- `ops/pre_market_dry_run.py` - Use real data for validation
- `ops/daily_paper_trading.py` - Real data freshness checks

### 3. 🔧 FIX SYSTEMD PERMISSION ISSUE

The systemd services are failing due to group permissions. Use cron instead for now:

```bash
./daily_paper_trading.sh setup    # Choose option 2 (cron)
```

Or fix systemd by removing the `User=` line from service files.

## 📊 ACTIVE MONITORING WORKFLOW (Don't Just Wait!)

### Daily Routine (Central Time)

**Morning (08:00 CT):**
```bash
./daily_paper_trading.sh preflight
# Check: Data fresh? Features valid? System healthy?
```

**Market Hours (08:30-15:00 CT):**
```bash
./daily_paper_trading.sh status
# Monitor: Positions, PnL, alerts, entropy
```

**End of Day (15:10 CT):**
```bash
./daily_paper_trading.sh eod
# Review: Daily IC, Sharpe, turnover, cost deviation
```

### Weekly Review Checklist

**Performance Gates (Must Stay Green):**
- ✅ **IC ≥ 0.015** (weekly average)
- ✅ **Sharpe ≥ 0.30** (net, after costs)
- ✅ **Turnover ≤ 2.0×/month**
- ✅ **≤1 guard breach per week**
- ✅ **Realized costs ≤ assumed +25%**

**Operational Health:**
- ✅ **Data freshness** (no stale feeds)
- ✅ **Position reconciliation** (matches broker)
- ✅ **No emergency halts** (or resolved quickly)
- ✅ **Feature whitelist integrity** (unchanged)

**Red Flags (Immediate Investigation):**
- 🚨 **IC < 0.010** (signal deterioration)
- 🚨 **Realized costs > assumed +50%** (execution issues)
- 🚨 **>5% orders blocked by ADV** (capacity issues)
- 🚨 **Position reconciliation failures** (broker integration)

## 📈 WHAT TO ADJUST (AND WHAT NOT TO TOUCH)

### ✅ ALLOWED ADJUSTMENTS (Operational Only)

**If turnover too high:**
```python
# Increase EWMA smoothing
smoothing_alpha = 0.2  # from 0.1
```

**If many ADV breaches:**
```python
# Reduce position size or max positions
max_participation_pct = 0.015  # from 0.02
```

**If slippage higher than expected:**
```python
# Increase impact coefficient (never lower!)
impact_coeff = 1.2  # from 1.0
```

### 🚫 FORBIDDEN CHANGES (Would Invalidate Validation)

- ❌ **Feature set** (locked whitelist)
- ❌ **Model parameters** (XGBoost config frozen)
- ❌ **Prediction logic** (no alpha tweaks)
- ❌ **Risk neutralization** (sector/size exposure)
- ❌ **Horizon or embargo** (temporal structure)

## 📊 MONITORING DASHBOARD

**Create simple monitoring script:**
```bash
# Check key metrics daily
./daily_paper_trading.sh status
tail -20 logs/daily_operations.log
ls -la results/paper/reports/daily_*.json | tail -5
```

**Weekly performance summary:**
```python
# ops/weekly_summary.py
def summarize_week():
    # Aggregate daily reports
    # Check against gates
    # Flag any issues
```

## 🎯 SUCCESS CRITERIA (20-Day Validation)

**Promotion to Live Trading Requirements:**

1. **Performance (ALL must pass):**
   - Paper IC ≥ 0.015 (period average)
   - Sharpe ≥ 0.30 net (period)
   - Turnover ≤ 2.0×/month (average)

2. **Operational (ALL must pass):**
   - ≤1 guard breach/week (total ≤4 in 20 days)
   - Realized costs ≤ assumed +25% (median)
   - Clean reconciliation ≥10 consecutive days
   - No unresolved emergency halts

3. **System Health (ALL must pass):**
   - Feature whitelist integrity maintained
   - Data feeds reliable (>95% uptime)
   - Rollback drill re-passed during period
   - All CI checks green throughout

## 🚨 FAILURE CONDITIONS (Abort Paper Trading)

**Immediate abort if:**
- IC < 0.005 for 5 consecutive days
- Sharpe net < 0.15 for 1 week
- >3 emergency halts in 1 week
- Position reconciliation fails >2 days
- Feature whitelist compromised

**Investigation required if:**
- IC trending downward (>-0.001/day for 5 days)
- Costs trending upward (>+5% per week)
- Entropy floor triggered >2×/week
- ADV breaches >10%/day

## 📋 YOUR ACTION ITEMS

### This Week:
1. **Integrate real data provider** (yfinance or better)
2. **Fix systemd permissions** or switch to cron
3. **Run first real dry-run** with actual market data
4. **Set up daily monitoring routine**

### Ongoing (20 Days):
1. **Check gates daily** (performance + operational)
2. **Review weekly summaries** (trends + red flags)
3. **Adjust only operational knobs** (no alpha changes)
4. **Document any issues** (for promotion decision)

### After 20 Days:
1. **Evaluate against promotion criteria**
2. **Make go/no-go decision** for live trading
3. **Document lessons learned**
4. **Scale testing** (if promoted)

## 🎉 BOTTOM LINE

**You're not passively waiting - you're actively validating!** Paper trading is about proving your system works with real market conditions, not just mock data.

**Priority 1:** Get real data integrated
**Priority 2:** Establish daily monitoring rhythm  
**Priority 3:** Track weekly against your gates

**This is professional validation, not a waiting period!** 🚀
