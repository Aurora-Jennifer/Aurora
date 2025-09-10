#!/bin/bash
echo "📊 EXECUTION-ENABLED PAPER TRADING STATUS"
echo "========================================"

echo ""
echo "⏰ TIMER STATUS:"
systemctl --user list-timers paper-trading-session.timer || true

echo ""
echo "🔧 SERVICE STATUS:"
systemctl --user status paper-trading-session.service --no-pager -l || true

echo ""
echo "📋 RECENT EXECUTION LOGS (last 20 lines):"
journalctl --user -u paper-trading-session.service --no-pager -n 20 | grep -E "(EXECUTION|ORDER|RISK|PORTFOLIO|ALPACA)" || true

echo ""
echo "🔍 EXECUTION HEALTH CHECK:"
if [ -f "ops/daily_paper_trading_with_execution.py" ]; then
    echo "   ✅ Execution-enabled script available"
else
    echo "   ❌ Execution-enabled script not found"
fi

if [ -f "config/execution.yaml" ]; then
    echo "   ✅ Execution configuration available"
else
    echo "   ❌ Execution configuration not found"
fi

if [ -f "$HOME/.config/paper-trading.env" ]; then
    echo "   ✅ Alpaca credentials configured"
else
    echo "   ⚠️  Alpaca credentials not configured"
fi
