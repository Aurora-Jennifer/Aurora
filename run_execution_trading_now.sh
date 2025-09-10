#!/bin/bash
echo "🔄 RUNNING EXECUTION-ENABLED PAPER TRADING MANUALLY"
echo "=================================================="

cd /home/Jennifer/secure/trader

# Set environment
export PYTHONPATH="$(pwd)"
export EXECUTION_MODE=paper
export EXECUTION_CONFIG_PATH="$(pwd)/config/execution.yaml"

# Load Alpaca credentials
if [ -f "$HOME/.config/paper-trading.env" ]; then
    set -a
    source "$HOME/.config/paper-trading.env"
    set +a
    echo "✅ Alpaca credentials loaded"
else
    echo "⚠️  No Alpaca credentials found"
    exit 1
fi

# Run trading session with execution
python3 ops/daily_paper_trading_with_execution.py --mode trading

echo ""
echo "✅ Manual execution run completed"
echo "📊 Check your Alpaca dashboard for executed orders"
