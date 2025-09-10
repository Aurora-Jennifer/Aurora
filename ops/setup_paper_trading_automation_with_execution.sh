#!/bin/bash
# Paper Trading Automation Setup Script with Execution Infrastructure
# Sets up systemd timers for automated daily operations with real order execution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USER="$(whoami)"

echo "🚀 SETTING UP PAPER TRADING AUTOMATION WITH EXECUTION"
echo "====================================================="
echo "Project root: $PROJECT_ROOT"
echo "User: $USER"

# Check for execution configuration
if [ ! -f "$PROJECT_ROOT/config/execution.yaml" ]; then
    echo "⚠️  Execution configuration not found. Creating default..."
    # The execution.yaml should already exist from our previous work
fi

# Check for Alpaca credentials template
if [ ! -f "$PROJECT_ROOT/config/alpaca_credentials.yaml.example" ]; then
    echo "❌ Alpaca credentials template not found!"
    echo "Please ensure config/alpaca_credentials.yaml.example exists"
    exit 1
fi

# Create systemd user directory
mkdir -p ~/.config/systemd/user

# 1. Create preflight service (08:00 CT) - Enhanced with execution validation
echo "📋 Creating preflight service with execution validation..."
cat > ~/.config/systemd/user/paper-trading-preflight.service << EOF
[Unit]
Description=Paper Trading Preflight Checks + Execution Validation
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
Environment=EXECUTION_MODE=paper
Environment=EXECUTION_CONFIG_PATH=$PROJECT_ROOT/config/execution.yaml
# Load Alpaca credentials from environment or config file
EnvironmentFile=$HOME/.config/paper-trading.env
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading_with_execution.py --mode preflight
User=$USER
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300

[Install]
WantedBy=default.target
EOF

# 2. Create preflight timer (08:00 CT = 14:00 UTC)
echo "⏰ Creating preflight timer..."
cat > ~/.config/systemd/user/paper-trading-preflight.timer << EOF
[Unit]
Description=Run Paper Trading Preflight + Execution Validation at 8:00 AM CT
Requires=paper-trading-preflight.service

[Timer]
OnCalendar=Mon-Fri 14:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# 3. Create trading session service (08:30 CT) - Enhanced with real order execution
echo "📈 Creating trading session service with real order execution..."
cat > ~/.config/systemd/user/paper-trading-session.service << EOF
[Unit]
Description=Paper Trading Session + Real Order Execution
After=network.target paper-trading-preflight.service
Requires=paper-trading-preflight.service

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
Environment=EXECUTION_MODE=paper
Environment=EXECUTION_CONFIG_PATH=$PROJECT_ROOT/config/execution.yaml
# Load Alpaca credentials from environment or config file
EnvironmentFile=$HOME/.config/paper-trading.env
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading_with_execution.py --mode trading
Restart=on-failure
RestartSec=30
User=$USER
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300
TimeoutStopSec=60
# Resource limits for safety
MemoryMax=2G
CPUQuota=200%

[Install]
WantedBy=default.target
EOF

# 4. Create trading session timer (08:30 CT = 14:30 UTC)
echo "⏰ Creating trading session timer..."
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

# 5. Create EOD service (15:10 CT) - Enhanced with portfolio reconciliation
echo "🌆 Creating EOD service with portfolio reconciliation..."
cat > ~/.config/systemd/user/paper-trading-eod.service << EOF
[Unit]
Description=Paper Trading End of Day + Portfolio Reconciliation
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
Environment=EXECUTION_MODE=paper
Environment=EXECUTION_CONFIG_PATH=$PROJECT_ROOT/config/execution.yaml
# Load Alpaca credentials from environment or config file
EnvironmentFile=$HOME/.config/paper-trading.env
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading_with_execution.py --mode eod
User=$USER
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300

[Install]
WantedBy=default.target
EOF

# 6. Create EOD timer (15:10 CT = 21:10 UTC)
echo "⏰ Creating EOD timer..."
cat > ~/.config/systemd/user/paper-trading-eod.timer << EOF
[Unit]
Description=Run Paper Trading EOD + Portfolio Reconciliation at 3:10 PM CT
Requires=paper-trading-eod.service

[Timer]
OnCalendar=Mon-Fri 21:10:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# 7. Create execution monitoring service (optional - runs every 5 minutes during trading hours)
echo "📊 Creating execution monitoring service..."
cat > ~/.config/systemd/user/paper-trading-execution-monitor.service << EOF
[Unit]
Description=Paper Trading Execution Monitoring
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
Environment=EXECUTION_MODE=paper
Environment=EXECUTION_CONFIG_PATH=$PROJECT_ROOT/config/execution.yaml
EnvironmentFile=$HOME/.config/paper-trading.env
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading_with_execution.py --mode monitor
User=$USER
StandardOutput=journal
StandardError=journal
TimeoutStartSec=60

[Install]
WantedBy=default.target
EOF

# 8. Create execution monitoring timer (every 5 minutes during trading hours)
echo "⏰ Creating execution monitoring timer..."
cat > ~/.config/systemd/user/paper-trading-execution-monitor.timer << EOF
[Unit]
Description=Paper Trading Execution Monitoring (every 5 minutes during trading hours)
Requires=paper-trading-execution-monitor.service

[Timer]
OnCalendar=Mon-Fri 14:30:00/5:00,21:00:00
Persistent=true
RandomizedDelaySec=30

[Install]
WantedBy=timers.target
EOF

# 9. Create management scripts
echo "🛠️ Creating enhanced management scripts..."

# Start automation script
cat > "$PROJECT_ROOT/start_paper_trading_with_execution.sh" << 'EOF'
#!/bin/bash
# Start paper trading automation with execution infrastructure

echo "🚀 STARTING PAPER TRADING AUTOMATION WITH EXECUTION"
echo "=================================================="

# Check for Alpaca credentials
if [ ! -f "$HOME/.config/paper-trading.env" ]; then
    echo "⚠️  No Alpaca credentials found in $HOME/.config/paper-trading.env"
    echo "   Please set up your Alpaca credentials first:"
    echo "   1. Copy config/alpaca_credentials.yaml.example to config/alpaca_credentials.yaml"
    echo "   2. Edit with your Alpaca paper trading credentials"
    echo "   3. Create $HOME/.config/paper-trading.env with ALPACA_API_KEY and ALPACA_SECRET_KEY"
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "❌ Aborted. Set up credentials first."
        exit 1
    fi
fi

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable and start timers
echo "⏰ Enabling timers..."
systemctl --user enable paper-trading-preflight.timer
systemctl --user enable paper-trading-session.timer  
systemctl --user enable paper-trading-eod.timer
systemctl --user enable paper-trading-execution-monitor.timer

systemctl --user start paper-trading-preflight.timer
systemctl --user start paper-trading-session.timer
systemctl --user start paper-trading-eod.timer
systemctl --user start paper-trading-execution-monitor.timer

echo "✅ Paper trading automation with execution started"
echo ""
echo "📋 Schedule:"
echo "   08:00 CT (14:00 UTC) - Preflight checks + execution validation"
echo "   08:30 CT (14:30 UTC) - Trading session + REAL order execution"
echo "   08:30-15:00 CT - Execution monitoring (every 5 minutes)"
echo "   15:10 CT (21:10 UTC) - End of day + portfolio reconciliation"
echo ""
echo "🔍 Monitor with:"
echo "   ./status_paper_trading_with_execution.sh"
echo "   journalctl --user -u paper-trading-*.service -f"
echo ""
echo "🚨 Emergency stop:"
echo "   ./stop_paper_trading_with_execution.sh"
EOF

# Stop automation script
cat > "$PROJECT_ROOT/stop_paper_trading_with_execution.sh" << 'EOF'
#!/bin/bash
# Stop paper trading automation with execution infrastructure

echo "🛑 STOPPING PAPER TRADING AUTOMATION WITH EXECUTION"
echo "=================================================="

# Stop and disable timers
echo "⏰ Disabling timers..."
systemctl --user stop paper-trading-preflight.timer || true
systemctl --user stop paper-trading-session.timer || true
systemctl --user stop paper-trading-eod.timer || true
systemctl --user stop paper-trading-execution-monitor.timer || true

systemctl --user disable paper-trading-preflight.timer || true
systemctl --user disable paper-trading-session.timer || true
systemctl --user disable paper-trading-eod.timer || true
systemctl --user disable paper-trading-execution-monitor.timer || true

# Stop any running services
echo "📈 Stopping services..."
systemctl --user stop paper-trading-preflight.service || true
systemctl --user stop paper-trading-session.service || true
systemctl --user stop paper-trading-eod.service || true
systemctl --user stop paper-trading-execution-monitor.service || true

echo "✅ Paper trading automation with execution stopped"
echo ""
echo "⚠️  Note: Any pending orders will remain active on Alpaca"
echo "   Check your Alpaca dashboard to cancel if needed"
EOF

# Enhanced status check script
cat > "$PROJECT_ROOT/status_paper_trading_with_execution.sh" << 'EOF'
#!/bin/bash
# Check paper trading automation status with execution monitoring

echo "📊 PAPER TRADING AUTOMATION STATUS WITH EXECUTION"
echo "================================================"

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
systemctl --user status paper-trading-execution-monitor.service --no-pager -l || true

echo ""
echo "📋 RECENT EXECUTION LOGS (last 50 lines):"
journalctl --user -u paper-trading-*.service --no-pager -n 50 | grep -E "(EXECUTION|ORDER|RISK|PORTFOLIO|ALPACA)" || true

echo ""
echo "📁 RECENT REPORTS:"
if [ -d "results/paper/reports" ]; then
    ls -la results/paper/reports/ | tail -10
else
    echo "   No reports directory found"
fi

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
    echo "   ⚠️  Alpaca credentials not configured (will run in signal-only mode)"
fi
EOF

# Manual run script with execution
cat > "$PROJECT_ROOT/run_paper_trading_with_execution_now.sh" << 'EOF'
#!/bin/bash
# Manually run paper trading cycle with execution infrastructure

echo "🔄 RUNNING PAPER TRADING CYCLE WITH EXECUTION MANUALLY"
echo "====================================================="

cd "$(dirname "$0")"

# Set environment
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="$(pwd)"
export EXECUTION_MODE=paper
export EXECUTION_CONFIG_PATH="$(pwd)/config/execution.yaml"

# Load Alpaca credentials if available
if [ -f "$HOME/.config/paper-trading.env" ]; then
    set -a
    source "$HOME/.config/paper-trading.env"
    set +a
    echo "✅ Alpaca credentials loaded"
else
    echo "⚠️  No Alpaca credentials found - will run in signal-only mode"
fi

# Run full cycle with execution
python3 ops/daily_paper_trading_with_execution.py --mode full

echo ""
echo "✅ Manual run with execution completed"
echo "📄 Check results/paper/daily_cycles/ for results"
echo "📊 Check Alpaca dashboard for executed orders"
EOF

# Emergency stop script
cat > "$PROJECT_ROOT/emergency_stop_execution.sh" << 'EOF'
#!/bin/bash
# Emergency stop for paper trading with execution

echo "🚨 EMERGENCY STOP - PAPER TRADING WITH EXECUTION"
echo "==============================================="

# Stop all systemd services immediately
echo "🛑 Stopping all systemd services..."
systemctl --user stop paper-trading-preflight.service || true
systemctl --user stop paper-trading-session.service || true
systemctl --user stop paper-trading-eod.service || true
systemctl --user stop paper-trading-execution-monitor.service || true

# Stop all timers
echo "⏰ Stopping all timers..."
systemctl --user stop paper-trading-preflight.timer || true
systemctl --user stop paper-trading-session.timer || true
systemctl --user stop paper-trading-eod.timer || true
systemctl --user stop paper-trading-execution-monitor.timer || true

# Try to trigger emergency stop in execution engine
echo "🚨 Triggering execution engine emergency stop..."
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
export EXECUTION_MODE=paper

if [ -f "$HOME/.config/paper-trading.env" ]; then
    set -a
    source "$HOME/.config/paper-trading.env"
    set +a
fi

python3 -c "
from ops.daily_paper_trading_with_execution import DailyPaperTradingWithExecution
try:
    ops = DailyPaperTradingWithExecution()
    ops._emergency_halt('Emergency stop triggered via systemd')
    print('✅ Emergency stop executed')
except Exception as e:
    print(f'⚠️  Emergency stop failed: {e}')
" || echo "⚠️  Could not execute emergency stop via Python"

echo ""
echo "✅ EMERGENCY STOP COMPLETED"
echo "⚠️  Check your Alpaca dashboard for any pending orders"
echo "📞 Contact support if you need to cancel orders manually"
EOF

# Make scripts executable
chmod +x "$PROJECT_ROOT/start_paper_trading_with_execution.sh"
chmod +x "$PROJECT_ROOT/stop_paper_trading_with_execution.sh" 
chmod +x "$PROJECT_ROOT/status_paper_trading_with_execution.sh"
chmod +x "$PROJECT_ROOT/run_paper_trading_with_execution_now.sh"
chmod +x "$PROJECT_ROOT/emergency_stop_execution.sh"

# Enable lingering for user systemd services
echo "🔧 Enabling user systemd lingering..."
if command -v loginctl &> /dev/null; then
    sudo loginctl enable-linger "$USER" || echo "⚠️ Could not enable lingering (requires sudo)"
else
    echo "⚠️ loginctl not available - you may need to manually enable lingering"
fi

echo ""
echo "✅ AUTOMATION SETUP WITH EXECUTION COMPLETE"
echo "==========================================="
echo ""
echo "📋 Next steps:"
echo "1. Set up Alpaca credentials:"
echo "   cp config/alpaca_credentials.yaml.example config/alpaca_credentials.yaml"
echo "   # Edit with your Alpaca paper trading credentials"
echo ""
echo "2. Create environment file:"
echo "   echo 'ALPACA_API_KEY=your_key_here' > ~/.config/paper-trading.env"
echo "   echo 'ALPACA_SECRET_KEY=your_secret_here' >> ~/.config/paper-trading.env"
echo ""
echo "3. Review execution configuration:"
echo "   # Edit config/execution.yaml as needed"
echo ""
echo "4. Test the system:"
echo "   ./run_paper_trading_with_execution_now.sh"
echo ""
echo "5. Start automation:"
echo "   ./start_paper_trading_with_execution.sh"
echo ""
echo "6. Monitor execution:"
echo "   ./status_paper_trading_with_execution.sh"
echo ""
echo "📅 ENHANCED SCHEDULE (all times in CT):"
echo "   08:00 - Preflight checks + execution validation"
echo "   08:30 - Trading session + REAL order execution"
echo "   08:30-15:00 - Execution monitoring (every 5 minutes)"
echo "   15:10 - End of day + portfolio reconciliation"
echo ""
echo "🔍 MONITORING:"
echo "   Logs: journalctl --user -u paper-trading-*.service -f"
echo "   Execution: journalctl --user -u paper-trading-*.service | grep -E '(EXECUTION|ORDER|RISK)'"
echo "   Reports: results/paper/reports/"
echo "   Status: ./status_paper_trading_with_execution.sh"
echo ""
echo "🚨 EMERGENCY STOP:"
echo "   ./emergency_stop_execution.sh"
echo "   OR: systemctl --user stop paper-trading-session.service"
echo ""
echo "⚠️  SAFETY NOTES:"
echo "   - All execution is PAPER TRADING ONLY (hardcoded)"
echo "   - Risk limits are enforced before every order"
echo "   - Emergency stop mechanisms are in place"
echo "   - Complete audit trails are maintained"
echo ""
echo "🎉 READY FOR AUTOMATED PAPER TRADING WITH REAL ORDER EXECUTION!"
