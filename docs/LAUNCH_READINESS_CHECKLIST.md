# Launch Readiness Checklist

## Pre-Launch Validation (Day -1)

### ✅ Data Pipeline
- [ ] **Data Coverage**: 130/130 symbols in `data/latest/prices.parquet`
- [ ] **Data Freshness**: Latest trading day data available
- [ ] **Symbol Normalization**: All symbols compatible with Alpaca API
- [ ] **Mock Fallback**: Disabled in production (strict mode)
- [ ] **Batching**: Large symbol sets processed correctly

### ✅ Feature Engineering
- [ ] **Cross-Sectional Dispersion**: Features have proper cross-sectional variation
- [ ] **Whitelist**: 45 protected features loaded and validated
- [ ] **Sector Snapshot**: `snapshots/sector_map.parquet` exists and current
- [ ] **Residualization**: Safe residualization against market/sector factors
- [ ] **Date Alignment**: No lookahead bias in feature construction

### ✅ Model Training
- [ ] **Leakage Audit**: OOF IC < 0.1 (no structural leakage)
- [ ] **Feature Allowlist**: Only approved features used in training
- [ ] **Deterministic Training**: Fixed random seeds and device consistency
- [ ] **Early Stopping**: Models trained with proper validation
- [ ] **Device Consistency**: Training and prediction on same device

### ✅ Risk Management
- [ ] **Market Neutrality**: Beta ≈ 0 across all positions
- [ ] **Sector Limits**: Sector exposure within defined bounds
- [ ] **Capacity Constraints**: ADV-based position limits enforced
- [ ] **Slippage Model**: Volume-dependent cost model implemented
- [ ] **Position Sizing**: Monotone score-to-weight mapping

### ✅ Automation
- [ ] **Systemd Services**: All services enabled and scheduled
- [ ] **Environment Variables**: Persistent config in `~/.config/paper-trading.env`
- [ ] **User Lingering**: Enabled for background execution
- [ ] **Timer Schedules**: Proper timing for market hours
- [ ] **Logging**: Structured logs with proper encoding

### ✅ Monitoring
- [ ] **Enhanced Dry Run**: Passes with data coverage gate
- [ ] **Kill Switch**: `kill.flag` mechanism tested
- [ ] **Alert System**: Notifications configured for critical events
- [ ] **Performance Metrics**: Baseline metrics established
- [ ] **Rollback Plan**: Tagged release with revert capability

## Launch Day Operations (Day 0)

### 08:00 CT - Pre-Market
- [ ] **Preflight Check**: `./monitor_paper_trading.sh` shows green status
- [ ] **Data Validation**: 130/130 symbols confirmed
- [ ] **Feature Pipeline**: All 45 features present and valid
- [ ] **Model Loading**: Trained model loads without errors
- [ ] **Risk Limits**: All risk parameters within bounds

### 08:05 CT - Trading Start
- [ ] **Position Generation**: Long/short positions calculated
- [ ] **Risk Checks**: Market beta, sector exposure validated
- [ ] **Capacity Limits**: ADV constraints enforced
- [ ] **Order Generation**: Paper orders created and logged
- [ ] **Execution**: Orders submitted to paper trading system

### Intraday Monitoring
- [ ] **Status Checks**: Hourly system health monitoring
- [ ] **Performance Tracking**: P&L and exposure monitoring
- [ ] **Alert Response**: Address any critical alerts
- [ ] **Data Updates**: Ensure data pipeline continues running
- [ ] **Log Review**: Check for any errors or anomalies

### 15:15 CT - End of Day
- [ ] **EOD Report**: Performance summary generated
- [ ] **Position Reconciliation**: All positions properly closed
- [ ] **P&L Calculation**: Daily performance calculated
- [ ] **Risk Metrics**: Final risk exposure reported
- [ ] **Next Day Prep**: Data fetch for tomorrow initiated

## Post-Launch Validation (Day +1)

### Performance Review
- [ ] **IC Analysis**: Information coefficient within expected range
- [ ] **Sharpe Ratio**: Risk-adjusted returns meet targets
- [ ] **Drawdown**: Maximum drawdown within limits
- [ ] **Turnover**: Trading activity within expected bounds
- [ ] **Costs**: Slippage and fees as expected

### System Health
- [ ] **Data Quality**: All symbols have complete data
- [ ] **Feature Stability**: Feature distributions stable
- [ ] **Model Performance**: Predictions show proper dispersion
- [ ] **Risk Controls**: All limits properly enforced
- [ ] **Automation**: All scheduled tasks completed successfully

### Operational Metrics
- [ ] **Uptime**: System available for full trading day
- [ ] **Latency**: Order execution within acceptable timeframes
- [ ] **Accuracy**: Position calculations match expectations
- [ ] **Reliability**: No critical errors or system failures
- [ ] **Scalability**: System handles current load efficiently

## Emergency Procedures

### Kill Switch Activation
```bash
# Emergency halt (within 60 seconds)
touch kill.flag

# Verify halt
./monitor_paper_trading.sh
```

### Data Issues
```bash
# Check data coverage
python -c "
import pandas as pd
df = pd.read_parquet('data/latest/prices.parquet')
print(f'Coverage: {df.symbol.nunique()}/130 symbols')
"

# Re-fetch data if needed
./daily_paper_trading.sh fetch
```

### Service Issues
```bash
# Check service status
systemctl --user status paper-*

# View logs
journalctl --user -u paper-* -f

# Restart services if needed
systemctl --user restart paper-preflight.service
```

### Performance Issues
```bash
# Check system resources
htop

# Review recent logs
journalctl --user -u paper-* --since "1 hour ago"

# Verify kill switch status
ls -la kill.flag
```

## Success Criteria

### Day 1 Targets
- **Data Coverage**: 100% (130/130 symbols)
- **Feature Count**: 45 features present
- **IC Range**: -0.1 to +0.1 (no leakage)
- **Sharpe Ratio**: > 0.5 (if positive)
- **Max Drawdown**: < 5%
- **Uptime**: > 95% of trading hours

### Week 1 Targets
- **Average IC**: > 0.05 (if positive)
- **Sharpe Ratio**: > 1.0 (if positive)
- **Max Drawdown**: < 10%
- **Turnover**: < 2.0 (monthly)
- **Cost Ratio**: < 0.5% (monthly)
- **Uptime**: > 99% of trading hours

### Month 1 Targets
- **Average IC**: > 0.08 (if positive)
- **Sharpe Ratio**: > 1.5 (if positive)
- **Max Drawdown**: < 15%
- **Win Rate**: > 55%
- **Cost Efficiency**: < 0.3% (monthly)
- **System Reliability**: > 99.5% uptime

## Rollback Plan

### Immediate Rollback (< 1 hour)
```bash
# Stop all services
systemctl --user stop paper-*

# Revert to previous tag
git checkout paper-launch-d0

# Restart services
systemctl --user start paper-*
```

### Full Rollback (< 4 hours)
```bash
# Complete system reset
git reset --hard paper-launch-d0
pip install -r requirements-lock.txt

# Rebuild data
./daily_paper_trading.sh fetch

# Restart automation
systemctl --user restart paper-*
```

### Data Recovery
```bash
# Restore from backup
cp data/backup/prices.parquet data/latest/

# Re-validate
python ops/enhanced_dry_run.py
```

## Contact Information

### Emergency Contacts
- **System Admin**: [Your contact info]
- **Alpaca Support**: support@alpaca.markets
- **GitHub Issues**: https://github.com/Aurora-Jennifer/Aurora/issues

### Key Resources
- **Documentation**: `docs/` directory
- **Logs**: `logs/` directory and `journalctl --user -u paper-*`
- **Config**: `~/.config/paper-trading.env`
- **Monitoring**: `./monitor_paper_trading.sh`
