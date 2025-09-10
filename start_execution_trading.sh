#!/bin/bash
echo "🚀 STARTING EXECUTION-ENABLED PAPER TRADING"
echo "==========================================="

# Reload systemd
systemctl --user daemon-reload

# Enable and start timer
systemctl --user enable paper-trading-session.timer
systemctl --user start paper-trading-session.timer

echo "✅ Execution-enabled paper trading started"
echo "📅 Schedule: 08:30 CT (14:30 UTC) - Trading session with real order execution"
echo ""
echo "🔍 Monitor with:"
echo "   journalctl --user -u paper-trading-session.service -f"
echo "   ./status_execution_trading.sh"
