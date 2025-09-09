#!/bin/bash
# Check paper trading automation status

echo "📊 PAPER TRADING AUTOMATION STATUS"
echo "=================================="

echo ""
echo "⏰ TIMER STATUS:"
systemctl --user list-timers paper-trading-*.timer || true

echo ""
echo "🔧 SERVICE STATUS:"
systemctl --user status paper-trading-preflight.service --no-pager -l || true
echo ""
systemctl --user status paper-trading-session.service --no-pager -l || true 
echo ""
systemctl --user status paper-trading-eod.service --no-pager -l || true

echo ""
echo "📋 RECENT LOGS (last 50 lines):"
journalctl --user -u paper-trading-*.service --no-pager -n 50 || true

echo ""
echo "📁 RECENT REPORTS:"
if [ -d "results/paper/reports" ]; then
    ls -la results/paper/reports/ | tail -10
else
    echo "   No reports directory found"
fi
