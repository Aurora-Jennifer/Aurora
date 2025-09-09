# QUICK REFERENCE COMMANDS

## 🚀 LAUNCH COMMANDS

### Start 20-Day Validation
```bash
export IS_PAPER_TRADING=true
./daily_paper_trading.sh full
```

### Daily Operations
```bash
# Check system status
./daily_paper_trading.sh status

# Run preflight checks only  
./daily_paper_trading.sh preflight

# Check automation status
systemctl --user status paper-trading-*

# View recent logs
journalctl --user -u paper-trading-preflight -f
```

## 🔧 DIAGNOSTIC COMMANDS

### Test Data Providers
```bash
# Test yfinance (working)
python ml/real_data_provider.py

# Test Alpaca (debugging)
python scripts/quick_alpaca_test.py
python scripts/test_alpaca_integration.py
```

### Check System Health
```bash
# Feature pipeline test
python -c "from ml.panel_builder import build_panel_from_universe; print('✅ Pipeline OK')"

# Risk controls test  
python -c "from ml.capacity_enforcement import ADVCapacityEnforcer; print('✅ Risk controls OK')"

# Automation test
python ops/daily_paper_trading.py --mode=status
```

## 📊 MONITORING COMMANDS

### Performance Tracking
```bash
# View daily reports
ls -la results/paper/reports/

# Check latest dry run
cat results/dry_runs/dry_run_*.json

# Monitor automation logs
tail -f logs/daily_operations.log
```

### System Validation
```bash
# Check feature whitelist integrity
python -c "
import json
with open('results/production/features_whitelist.json') as f:
    features = json.load(f)['features']
print(f'✅ {len(features)} features protected')
"

# Validate environment locks
python -c "
import os
print('✅ Paper mode:', os.environ.get('IS_PAPER_TRADING', 'NOT SET'))
"
```

## 🔐 ALPACA DEBUGGING

### API Authentication Tests
```bash
# Test paper trading API
curl -sS https://paper-api.alpaca.markets/v2/account \
  -H "APCA-API-KEY-ID: PKQ9ZKNTB5HV9SNQ929E" \
  -H "APCA-API-SECRET-KEY: HaZ9FkKaXJdK1HFxp6Vr3449nMXUgPWvbyZhMpPn" \
  -D -

# Test market data API
curl -sS "https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Min&start=2025-09-06T14:00:00Z&end=2025-09-06T14:10:00Z&feed=iex" \
  -H "APCA-API-KEY-ID: PKQ9ZKNTB5HV9SNQ929E" \
  -H "APCA-API-SECRET-KEY: HaZ9FkKaXJdK1HFxp6Vr3449nMXUgPWvbyZhMpPn" \
  -D -
```

### Environment Setup
```bash
# Set Alpaca environment (when working)
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
export APCA_API_KEY_ID="PKQ9ZKNTB5HV9SNQ929E"  
export APCA_API_SECRET_KEY="HaZ9FkKaXJdK1HFxp6Vr3449nMXUgPWvbyZhMpPn"
```

## 🛠️ MAINTENANCE COMMANDS

### Automation Management
```bash
# Start automation timers
systemctl --user start paper-trading-preflight.timer
systemctl --user start paper-trading-session.timer
systemctl --user start paper-trading-eod.timer

# Stop automation timers
systemctl --user stop paper-trading-*.timer

# Restart automation
systemctl --user daemon-reload
systemctl --user restart paper-trading-*.timer
```

### Dependency Management
```bash
# Install missing dependencies
pip install pandas-market-calendars
pip install --upgrade websockets
pip install alpaca-trade-api

# Check key dependencies
python -c "import yfinance, pandas, numpy, xgboost; print('✅ Core deps OK')"
```

### Data Management
```bash
# Clean old reports
find results/paper/reports/ -name "*.json" -mtime +30 -delete

# Clean old logs
find logs/ -name "*.log*" -mtime +7 -delete

# Archive dry runs
mkdir -p results/dry_runs/archive
mv results/dry_runs/*.json results/dry_runs/archive/
```

## 🆘 EMERGENCY COMMANDS

### Kill Switch Tests
```bash
# Test rollback (chaos test)
python ops/rollback_chaos_test.py

# Manual kill switch
touch kill.flag
# (System should detect and halt automatically)
```

### Recovery Commands
```bash
# Reset to clean state
git stash
git checkout main
git pull origin main

# Restart automation
systemctl --user daemon-reload
systemctl --user restart paper-trading-*.timer

# Verify system health
./daily_paper_trading.sh status
```

## 📞 SUPPORT CONTACTS

### Alpaca Support
```bash
# Email: support@alpaca.markets
# Subject: Paper Trading API Authentication Issues
# Include: API test results, account details, error messages
```

### System Status
```bash
# Generate diagnostic report
python3 << 'EOF'
import json, os, subprocess
status = {
    'environment': dict(os.environ),
    'git_status': subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout,
    'automation': subprocess.run(['systemctl', '--user', 'list-timers', 'paper-trading-*'], capture_output=True, text=True).stdout
}
print(json.dumps(status, indent=2))
EOF
```

---

**TIP:** Keep this file open in a separate tab for quick command reference during operations.
