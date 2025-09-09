#!/bin/bash
# Start paper trading automation

echo "🚀 STARTING PAPER TRADING AUTOMATION"
echo "===================================="

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable and start timers
echo "⏰ Enabling timers..."
systemctl --user enable paper-trading-preflight.timer
systemctl --user enable paper-trading-session.timer  
systemctl --user enable paper-trading-eod.timer

systemctl --user start paper-trading-preflight.timer
systemctl --user start paper-trading-session.timer
systemctl --user start paper-trading-eod.timer

echo "✅ Paper trading automation started"
echo ""
echo "📋 Schedule:"
echo "   08:00 CT (14:00 UTC) - Preflight checks"
echo "   08:30 CT (14:30 UTC) - Trading session"  
echo "   15:10 CT (21:10 UTC) - End of day operations"
echo ""
echo "🔍 Monitor with:"
echo "   ./status_paper_trading.sh"
echo "   journalctl --user -u paper-trading-*.service -f"
