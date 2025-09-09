#!/bin/bash
# Cron wrapper for end-of-day operations

# Set working directory
cd "/home/Jennifer/secure/trader"

# Set environment
export PATH="/usr/local/bin:/usr/bin:/bin"
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="/home/Jennifer/secure/trader"
export PYTHONHASHSEED=42
export PYTHONIOENCODING=utf-8

# Ensure log directory exists
mkdir -p logs

# Run with output redirection
exec python3 ops/daily_paper_trading.py --mode eod >> logs/cron_eod.log 2>&1
