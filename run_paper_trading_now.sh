#!/bin/bash
# Manually run paper trading cycle

echo "🔄 RUNNING PAPER TRADING CYCLE MANUALLY"
echo "======================================="

cd "$(dirname "$0")"

# Set environment
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="$(pwd)"

# Run full cycle
python3 ops/daily_paper_trading.py --mode full

echo ""
echo "✅ Manual run completed"
echo "📄 Check results/paper/daily_cycles/ for results"
