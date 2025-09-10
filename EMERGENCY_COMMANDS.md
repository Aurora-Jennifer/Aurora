# 🚨 EMERGENCY COMMANDS

## Kill Switch (Immediate Halt)
```bash
# Activate kill switch and restart services
export KILL_SWITCH=1 && systemctl --user restart paper-trading-session.service

# Check if kill switch is active
echo "Kill switch: ${KILL_SWITCH:-0}"
```

## Revert Kill Switch
```bash
# Remove kill switch and restart services
unset KILL_SWITCH && systemctl --user restart paper-trading-session.service
```

## Emergency Stop All Trading
```bash
# Stop all trading services
systemctl --user stop paper-trading-session.service
systemctl --user stop paper-trading-session.timer
systemctl --user stop paper-trading-execution-monitor.service
systemctl --user stop paper-trading-execution-monitor.timer
```

## Check Service Status
```bash
# Check if services are running
systemctl --user status paper-trading-session.service
systemctl --user status paper-trading-execution-monitor.service

# Check logs
journalctl --user -u paper-trading-session.service -f
```

## Manual Position Flattening (if needed)
```bash
# Run emergency position flattening
python3 ops/daily_paper_trading_with_execution.py --mode emergency-flatten
```

## Quick Health Check
```bash
# Check account status
python3 -c "
from alpaca.trading.client import TradingClient
import os
client = TradingClient(api_key=os.getenv('APCA_API_KEY_ID'), secret_key=os.getenv('APCA_API_SECRET_KEY'), paper=True)
account = client.get_account()
positions = client.get_all_positions()
print(f'Account: {account.id}')
print(f'Cash: \${float(account.cash):,.2f}')
print(f'Positions: {len(positions)}')
for pos in positions:
    print(f'  {pos.symbol}: {pos.qty} shares')
"
```

---
**Last Updated**: $(date)
**Tag**: execution-ready-$(date +%Y%m%d)
**Config SHA256**: e0268a08fdc125be0f1b8d92ad66a9ce741afe2b60403454d77265c7519dbcda
