# 📊 LOG FILTERS FOR TRADING MONITORING

## Quick Alert Search (Exclude MARKET_CLOSED)
```bash
# WARN/ERROR excluding MARKET_CLOSED
journalctl --user -u paper-trading-session.service | grep -E "(WARN|ERROR)" | grep -v "MARKET_CLOSED"

# Real-time monitoring (exclude MARKET_CLOSED)
journalctl --user -u paper-trading-session.service -f | grep -E "(WARN|ERROR)" | grep -v "MARKET_CLOSED"
```

## Signal Generation Monitoring
```bash
# Model usage and feature matching
journalctl --user -u paper-trading-session.service | grep -E "(USING MODEL|features matched|Signal distribution)"

# Signal generation success/failure
journalctl --user -u paper-trading-session.service | grep -E "(Generated.*signals|MODEL DISABLED)"
```

## Execution Monitoring
```bash
# Order submission and fills
journalctl --user -u paper-trading-session.service | grep -E "(Order submitted|Order filled|Execution result)"

# Pre-trade gate decisions
journalctl --user -u paper-trading-session.service | grep -E "(Order rejected|Gate adjusted|BLOCKED_)"
```

## Performance Monitoring
```bash
# Session summaries
journalctl --user -u paper-trading-session.service | grep -E "(TRADING SESSION SUMMARY|Duration:|Orders submitted)"

# Entropy and kill conditions
journalctl --user -u paper-trading-session.service | grep -E "(entropy=|kill condition|emergency halt)"
```

## Health Check Commands
```bash
# Service status
systemctl --user status paper-trading-session.service --no-pager

# Recent logs (last 50 lines)
journalctl --user -u paper-trading-session.service -n 50

# Logs from last hour
journalctl --user -u paper-trading-session.service --since "1 hour ago"
```

## Expected Good Logs
- ✅ "Execution infrastructure initialized successfully"
- ✅ "USING MODEL (45/45 features matched)"
- ✅ "Generated X trading signals using production model"
- ✅ "Signal distribution: mean=0.000, std=X.XXX, longs=20.0%, shorts=20.0%"
- ✅ "Order submitted successfully"
- ✅ "Pre-trade checks: X/X passed"

## Red Flag Logs
- ❌ "MODEL DISABLED: Error in prediction"
- ❌ "Order rejected by pre-trade gate"
- ❌ "BLOCKED_SELL_NO_POSITION"
- ❌ "Emergency halt triggered"
- ❌ "Kill condition triggered"
- ❌ "Feature contract violated"

---
**Usage**: Copy/paste these commands for quick log analysis during trading hours.
