#!/bin/bash
echo "🛑 STOPPING EXECUTION-ENABLED PAPER TRADING"
echo "==========================================="

# Stop timer and service
systemctl --user stop paper-trading-session.timer || true
systemctl --user stop paper-trading-session.service || true
systemctl --user disable paper-trading-session.timer || true

echo "✅ Execution-enabled paper trading stopped"
echo "⚠️  Check your Alpaca dashboard for any pending orders"
