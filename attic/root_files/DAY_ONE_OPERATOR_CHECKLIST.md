# DAY-1 OPERATOR CHECKLIST
*Paper Trading Launch Day*

## 🌅 PRE-MARKET (08:00 CT)

### ✅ MANDATORY PREFLIGHT CHECKS
```bash
export IS_PAPER_TRADING=true
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
./daily_paper_trading.sh preflight
```

**Expected Results:**
- ✅ Features: 45 (all whitelist features present)
- ✅ Symbols: ≥250 (sufficient universe)
- ✅ Date alignment: feature_date vs decision_date correct
- ✅ Sector snapshot: loaded and validated
- ✅ Risk controls: all systems operational
- ✅ Environment: paper trading mode confirmed

**RED FLAGS - HALT IF ANY:**
- ❌ Feature count ≠ 45
- ❌ Symbol count < 250
- ❌ Any preflight errors
- ❌ Real money environment detected

## 📊 MARKET HOURS (08:30-15:00 CT)

### Automated Monitoring Active
- **Systemd timers** handling operations
- **Real-time logging** to files
- **Automatic alerts** on anomalies

### Manual Status Checks (Hourly)
```bash
./daily_paper_trading.sh status
```

**Monitor For:**
- Position updates and signal changes
- ADV blocking (should be <5% of orders)
- Action entropy (should be >0.75)
- No reconciliation mismatches

### IMMEDIATE HALT TRIGGERS
- Daily PnL ≤ -2%
- Action entropy < 0.75 for 10 bars
- Position concentration breach
- System errors or crashes

**Emergency Command:**
```bash
touch kill.flag
# System auto-halts within 60 seconds
```

## 🌆 END-OF-DAY (15:10 CT)

### Automated EOD Report
```bash
./daily_paper_trading.sh eod
```

**Key Metrics to Review:**
- **Daily IC:** Target ≥ 0.015 (weekly average)
- **Net Sharpe:** Target ≥ 0.30 
- **Turnover:** Target ≤ 2.0×/month
- **Realized Slippage:** Target ≤ assumed +25%
- **ADV Blocks:** Target <5% of orders
- **Factor R²:** Monitor for drift from backtest

### Daily Reconciliation
- ✅ Positions match broker
- ✅ Cash balance correct
- ✅ No orphaned orders
- ✅ Logs complete and clean

## 📈 WEEK 1 SUCCESS CRITERIA

### Performance Gates (Must Pass)
- **IC ≥ 0.015** (weekly average)
- **Sharpe ≥ 0.30** (net after costs)
- **Turnover ≤ 2.0×/month**
- **≤1 guard breach total**
- **Realized costs ≤ assumed +25%**

### Operational Excellence
- **100% data uptime**
- **Clean position reconciliation**
- **No manual interventions required**
- **All automation working smoothly**

## 🔧 ALLOWED ADJUSTMENTS (Week 1 Only)

**IF blocked_order_pct > 5%:**
```bash
# Reduce ADV participation cap
# Current: 2.0% → try 1.5%
```

**IF turnover > 2.0×/month:**
```bash
# Increase EWMA smoothing
# More conservative position changes
```

**IF realized slippage > assumed + 25%:**
```bash
# Increase impact coefficients (ONLY upward)
# More conservative cost assumptions
```

## 🚫 STRICTLY FORBIDDEN

**NEVER adjust during validation:**
- ❌ Feature selection or engineering
- ❌ Model parameters or training
- ❌ Risk limits (downward)
- ❌ Universe definition
- ❌ Fundamental methodology

## 📞 ESCALATION CONTACTS

### Technical Issues
- **Repository:** Complete documentation and runbooks
- **Logs:** `logs/daily_operations.log`
- **Kill-switch:** `touch kill.flag`

### Broker/API Issues  
- **Alpaca Support:** support@alpaca.markets
- **Paper Trading:** Confirm paper mode active
- **API Limits:** Monitor rate limiting

### Decision Points
- **Continue/Halt:** Based on daily metrics
- **Adjustment Approval:** Only allowed knobs
- **Week 1 Assessment:** Go/no-go for Week 2

## 🎯 SUCCESS INDICATORS

### Green Lights ✅
- All preflight checks pass
- IC tracking toward weekly target
- Costs within expected range
- Automation running smoothly
- Clean reconciliation daily

### Warning Signs ⚠️
- Preflight issues (investigate)
- IC below target (monitor trend)
- High ADV blocking (reduce participation)
- Reconciliation discrepancies

### Red Flags 🚨
- Multiple preflight failures
- Consistent IC underperformance  
- Cost explosion vs assumptions
- System crashes or data outages
- Any live trading mode detection

## 📊 DAILY LOG TEMPLATE

**Date:** [YYYY-MM-DD]
**Market Session:** [Normal/Holiday/Half-day]

**Preflight (08:00):**
- Features: [ ] Pass/Fail
- Symbols: [ ] Pass/Fail  
- Systems: [ ] Pass/Fail

**Trading (08:30-15:00):**
- Alerts: [ ] None/Details
- Manual interventions: [ ] None/Details
- Performance: [ ] On-track/Issues

**EOD (15:10):**
- Daily IC: [value]
- Net Sharpe: [value] 
- Turnover: [value]
- Costs: [realized vs assumed]
- Reconciliation: [ ] Clean/Issues

**Decision:**
- [ ] Continue validation
- [ ] Adjust operations (specify)
- [ ] Escalate issues
- [ ] Halt validation

---

**🚀 READY FOR DAY-1 LAUNCH! PROFESSIONAL OPERATION BEGINS!**
