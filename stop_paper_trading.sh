#!/bin/bash
# Stop paper trading automation

echo "🛑 STOPPING PAPER TRADING AUTOMATION"
echo "===================================="

# Stop and disable timers
echo "⏰ Disabling timers..."
systemctl --user stop paper-trading-preflight.timer || true
systemctl --user stop paper-trading-session.timer || true
systemctl --user stop paper-trading-eod.timer || true

systemctl --user disable paper-trading-preflight.timer || true
systemctl --user disable paper-trading-session.timer || true
systemctl --user disable paper-trading-eod.timer || true

# Stop any running services
echo "📈 Stopping services..."
systemctl --user stop paper-trading-preflight.service || true
systemctl --user stop paper-trading-session.service || true
systemctl --user stop paper-trading-eod.service || true

echo "✅ Paper trading automation stopped"
