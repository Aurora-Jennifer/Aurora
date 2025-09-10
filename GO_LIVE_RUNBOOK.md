# 🚀 GO-LIVE RUNBOOK

## Pre-Market Checklist (08:00-09:30 CT)

### 08:00 CT - Preflight Check
```bash
# Run preflight checks
python3 ops/daily_paper_trading_with_execution.py --mode preflight
```
**Expected**: ✅ PASSED (4/4 checks)

### 08:30 CT - Systemd Timer Start
```bash
# Check timer status
systemctl --user status paper-trading-session.timer

# Manual start if needed
systemctl --user start paper-trading-session.timer
```
**Expected**: Timer active, waiting for 08:30 CT

## Market Open (09:30-09:31 CT)

### 09:31:00 CT - Engine Start
**CRITICAL**: Start engine **1 minute after market open** to skip opening auction volatility.

**Expected Logs**:
- ✅ "Execution infrastructure initialized successfully" (once)
- ✅ "USING MODEL (45/45 features matched)"
- ✅ "Generated X trading signals using production model"
- ✅ "Signal distribution: mean=0.000, std=X.XXX, longs=20.0%, shorts=20.0%"

**Red Flags**:
- ❌ "MODEL DISABLED: Error in prediction"
- ❌ "GATE_ERROR" (only MARKET_CLOSED before 09:30 is OK)
- ❌ Multiple initialization messages

## First Hour Monitoring (09:31-10:31 CT)

### Position Limits (Conservative)
- **Per Symbol**: 1-2% of equity (vs normal 5%)
- **Gross Exposure**: 10-15% of equity (vs normal 60%)
- **Daily Orders**: Max 50 (vs normal 200)

### Watch For
1. **Gate Pass Rate**: Should be >90% (only MARKET_CLOSED rejections before 09:30)
2. **Submit→ACK Latency**: <5 seconds
3. **Reconciliation**: Clean position updates
4. **Signal Quality**: Mean≈0, std≈0.6, 20% longs, 20% shorts

### Log Monitoring
```bash
# Real-time alerts (exclude MARKET_CLOSED)
journalctl --user -u paper-trading-session.service -f | grep -E "(WARN|ERROR)" | grep -v "MARKET_CLOSED"

# Signal generation
journalctl --user -u paper-trading-session.service -f | grep -E "(USING MODEL|Signal distribution)"
```

## Normal Operations (10:31+ CT)

### Position Limits (Normal)
- **Per Symbol**: 5% of equity
- **Gross Exposure**: 60% of equity
- **Daily Orders**: Max 200

### Extended Hours (Optional)
If needed, enable extended hours trading:
```bash
# Edit config/execution.yaml
allow_extended_hours: true
```

## Emergency Procedures

### Kill Switch
```bash
# Immediate halt
export KILL_SWITCH=1 && systemctl --user restart paper-trading-session.service
```

### Revert Kill Switch
```bash
# Resume trading
unset KILL_SWITCH && systemctl --user restart paper-trading-session.service
```

### Emergency Stop
```bash
# Stop all services
systemctl --user stop paper-trading-session.service
systemctl --user stop paper-trading-session.timer
```

## End of Day (15:00+ CT)

### EOD Check
```bash
# Run EOD reporting
python3 ops/daily_paper_trading_with_execution.py --mode eod
```

### Account Reconciliation
```bash
# Check final positions
python3 -c "
from alpaca.trading.client import TradingClient
import os
client = TradingClient(api_key=os.getenv('APCA_API_KEY_ID'), secret_key=os.getenv('APCA_API_SECRET_KEY'), paper=True)
account = client.get_account()
positions = client.get_all_positions()
print(f'Final Portfolio Value: \${float(account.portfolio_value):,.2f}')
print(f'Positions: {len(positions)}')
"
```

## Success Metrics

### Day 1 Goals
- ✅ Model loads and generates signals
- ✅ Orders execute without errors
- ✅ Position tracking accurate
- ✅ No emergency halts
- ✅ Clean logs (no red flags)

### Week 1 Goals
- ✅ Consistent signal generation
- ✅ Order fill rate >95%
- ✅ Position reconciliation accuracy >99%
- ✅ Risk limits respected
- ✅ Performance tracking

---
**Last Updated**: $(date)
**Tag**: execution-ready-$(date +%Y%m%d)
**Config SHA256**: e0268a08fdc125be0f1b8d92ad66a9ce741afe2b60403454d77265c7519dbcda
