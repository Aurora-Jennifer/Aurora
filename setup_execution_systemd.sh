#!/bin/bash
# Quick setup script for execution-enabled systemd services

set -euo pipefail

echo "🚀 SETTING UP EXECUTION-ENABLED SYSTEMD SERVICES"
echo "================================================"

# Check if credentials are available
if [ ! -f "$HOME/.config/paper-trading.env" ]; then
    echo "❌ No credentials found at $HOME/.config/paper-trading.env"
    echo "Please ensure your Alpaca credentials are configured there"
    exit 1
fi

echo "✅ Found credentials file: $HOME/.config/paper-trading.env"

# Create systemd user directory
mkdir -p ~/.config/systemd/user

# Create the main trading service with execution
echo "📋 Creating paper-trading-session.service with execution..."
cat > ~/.config/systemd/user/paper-trading-session.service << EOF
[Unit]
Description=Paper Trading Session + Real Order Execution
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/Jennifer/secure/trader
Environment=PYTHONPATH=/home/Jennifer/secure/trader
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
Environment=EXECUTION_MODE=paper
Environment=EXECUTION_CONFIG_PATH=/home/Jennifer/secure/trader/config/execution.yaml
EnvironmentFile=$HOME/.config/paper-trading.env
ExecStart=/usr/bin/python3 /home/Jennifer/secure/trader/ops/daily_paper_trading_with_execution.py --mode trading
Restart=on-failure
RestartSec=30
User=$USER
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300
TimeoutStopSec=60
MemoryMax=2G
CPUQuota=200%

[Install]
WantedBy=default.target
EOF

# Create the timer
echo "⏰ Creating paper-trading-session.timer..."
cat > ~/.config/systemd/user/paper-trading-session.timer << EOF
[Unit]
Description=Run Paper Trading Session + Real Order Execution at 8:30 AM CT
Requires=paper-trading-session.service

[Timer]
OnCalendar=Mon-Fri 14:30:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# Create management scripts
echo "🛠️ Creating management scripts..."

# Start script
cat > /home/Jennifer/secure/trader/start_execution_trading.sh << 'EOF'
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
EOF

# Stop script
cat > /home/Jennifer/secure/trader/stop_execution_trading.sh << 'EOF'
#!/bin/bash
echo "🛑 STOPPING EXECUTION-ENABLED PAPER TRADING"
echo "==========================================="

# Stop timer and service
systemctl --user stop paper-trading-session.timer || true
systemctl --user stop paper-trading-session.service || true
systemctl --user disable paper-trading-session.timer || true

echo "✅ Execution-enabled paper trading stopped"
echo "⚠️  Check your Alpaca dashboard for any pending orders"
EOF

# Status script
cat > /home/Jennifer/secure/trader/status_execution_trading.sh << 'EOF'
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
EOF

# Manual run script
cat > /home/Jennifer/secure/trader/run_execution_trading_now.sh << 'EOF'
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
EOF

# Make scripts executable
chmod +x /home/Jennifer/secure/trader/start_execution_trading.sh
chmod +x /home/Jennifer/secure/trader/stop_execution_trading.sh
chmod +x /home/Jennifer/secure/trader/status_execution_trading.sh
chmod +x /home/Jennifer/secure/trader/run_execution_trading_now.sh

# Enable lingering
echo "🔧 Enabling user systemd lingering..."
if command -v loginctl &> /dev/null; then
    sudo loginctl enable-linger "$USER" || echo "⚠️ Could not enable lingering (requires sudo)"
fi

echo ""
echo "✅ EXECUTION-ENABLED SYSTEMD SETUP COMPLETE"
echo "==========================================="
echo ""
echo "📋 Your system is now configured for automated paper trading with real order execution!"
echo ""
echo "🎯 What happens at 08:30 CT (14:30 UTC) every weekday:"
echo "   1. System generates XGBoost trading signals"
echo "   2. Calculates position sizes based on risk limits"
echo "   3. Executes REAL orders on Alpaca (paper trading)"
echo "   4. Monitors order fills and portfolio updates"
echo "   5. Enforces risk limits and safety checks"
echo ""
echo "🚀 To start automated trading:"
echo "   ./start_execution_trading.sh"
echo ""
echo "🔍 To monitor execution:"
echo "   ./status_execution_trading.sh"
echo "   journalctl --user -u paper-trading-session.service -f"
echo ""
echo "🧪 To test manually:"
echo "   ./run_execution_trading_now.sh"
echo ""
echo "🛑 To stop:"
echo "   ./stop_execution_trading.sh"
echo ""
echo "⚠️  SAFETY NOTES:"
echo "   - All execution is PAPER TRADING ONLY"
echo "   - Risk limits are enforced before every order"
echo "   - Emergency stop mechanisms are in place"
echo "   - Complete audit trails are maintained"
echo ""
echo "🎉 READY FOR AUTOMATED PAPER TRADING WITH REAL ORDER EXECUTION!"
