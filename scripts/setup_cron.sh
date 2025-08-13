#!/bin/bash
"""
Cron Setup Script for Trading System
Sets up automated daily trading at 8:30 AM CT (Chicago time).
"""

set -e

# Configuration
REPO_PATH=$(pwd)
PYTHON_PATH=$(which python)
VENV_PATH="$REPO_PATH/.venv"
CRON_TIME="30 8 * * 1-5"  # 8:30 AM CT, weekdays only
IB_CLIENT_ID=4242

echo "🚀 Setting up automated trading cron job..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "enhanced_paper_trading.py" ]; then
    echo "❌ Error: Please run this script from the trading system root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "⚠️  Virtual environment not found at $VENV_PATH"
    echo "   Creating virtual environment..."
    python -m venv "$VENV_PATH"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing dependencies..."
source "$VENV_PATH/bin/activate"
pip install -e .

# Create the cron command
CRON_COMMAND="$CRON_TIME cd $REPO_PATH && source $VENV_PATH/bin/activate && IB_CLIENT_ID=$IB_CLIENT_ID $PYTHON_PATH enhanced_paper_trading.py --daily --profile config/live_profile.json >> logs/run_\$(date +\%F).log 2>&1"

echo "📅 Cron job configuration:"
echo "   Time: $CRON_TIME (8:30 AM CT, weekdays)"
echo "   Command: $CRON_COMMAND"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "enhanced_paper_trading.py"; then
    echo "⚠️  Cron job already exists. Current crontab:"
    crontab -l
    echo ""
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cron setup cancelled"
        exit 1
    fi
fi

# Add the cron job
echo "➕ Adding cron job..."
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo "✅ Cron job added successfully!"
echo ""
echo "📋 To verify the cron job:"
echo "   crontab -l"
echo ""
echo "📋 To edit the cron job:"
echo "   crontab -e"
echo ""
echo "📋 To remove the cron job:"
echo "   crontab -r"
echo ""
echo "📋 To check cron logs:"
echo "   tail -f /var/log/cron"
echo ""
echo "📋 To test the command manually:"
echo "   cd $REPO_PATH && source $VENV_PATH/bin/activate && IB_CLIENT_ID=$IB_CLIENT_ID $PYTHON_PATH enhanced_paper_trading.py --daily --profile config/live_profile.json"
echo ""
echo "🎉 Automated trading setup complete!"
echo "   The system will run daily at 8:30 AM CT on weekdays."
