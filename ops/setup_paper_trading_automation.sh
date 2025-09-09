#!/bin/bash
# Paper Trading Automation Setup Script
# Sets up systemd timers for automated daily operations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USER="$(whoami)"

echo "🚀 SETTING UP PAPER TRADING AUTOMATION"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "User: $USER"

# Create systemd user directory
mkdir -p ~/.config/systemd/user

# 1. Create preflight service (08:00 CT)
echo "📋 Creating preflight service..."
cat > ~/.config/systemd/user/paper-trading-preflight.service << EOF
[Unit]
Description=Paper Trading Preflight Checks
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading.py --mode preflight
User=$USER
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

# 2. Create preflight timer (08:00 CT = 14:00 UTC)
echo "⏰ Creating preflight timer..."
cat > ~/.config/systemd/user/paper-trading-preflight.timer << EOF
[Unit]
Description=Run Paper Trading Preflight at 8:00 AM CT
Requires=paper-trading-preflight.service

[Timer]
OnCalendar=Mon-Fri 14:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# 3. Create trading session service (08:30 CT)
echo "📈 Creating trading session service..."
cat > ~/.config/systemd/user/paper-trading-session.service << EOF
[Unit]
Description=Paper Trading Session
After=network.target paper-trading-preflight.service

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading.py --mode trading
Restart=on-failure
RestartSec=30
User=$USER
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300
TimeoutStopSec=60

[Install]
WantedBy=default.target
EOF

# 4. Create trading session timer (08:30 CT = 14:30 UTC)
echo "⏰ Creating trading session timer..."
cat > ~/.config/systemd/user/paper-trading-session.timer << EOF
[Unit]
Description=Run Paper Trading Session at 8:30 AM CT
Requires=paper-trading-session.service

[Timer]
OnCalendar=Mon-Fri 14:30:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# 5. Create EOD service (15:10 CT)
echo "🌆 Creating EOD service..."
cat > ~/.config/systemd/user/paper-trading-eod.service << EOF
[Unit]
Description=Paper Trading End of Day Operations
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=IS_PAPER_TRADING=true
Environment=TRADING_TIMEZONE=America/Chicago
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading.py --mode eod
User=$USER
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

# 6. Create EOD timer (15:10 CT = 21:10 UTC)
echo "⏰ Creating EOD timer..."
cat > ~/.config/systemd/user/paper-trading-eod.timer << EOF
[Unit]
Description=Run Paper Trading EOD at 3:10 PM CT
Requires=paper-trading-eod.service

[Timer]
OnCalendar=Mon-Fri 21:10:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# 7. Create management scripts
echo "🛠️ Creating management scripts..."

# Start automation script
cat > "$PROJECT_ROOT/start_paper_trading.sh" << 'EOF'
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
EOF

# Stop automation script
cat > "$PROJECT_ROOT/stop_paper_trading.sh" << 'EOF'
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
EOF

# Status check script
cat > "$PROJECT_ROOT/status_paper_trading.sh" << 'EOF'
#!/bin/bash
# Check paper trading automation status

echo "📊 PAPER TRADING AUTOMATION STATUS"
echo "=================================="

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
echo "📋 RECENT LOGS (last 50 lines):"
journalctl --user -u paper-trading-*.service --no-pager -n 50 || true

echo ""
echo "📁 RECENT REPORTS:"
if [ -d "results/paper/reports" ]; then
    ls -la results/paper/reports/ | tail -10
else
    echo "   No reports directory found"
fi
EOF

# Manual run script
cat > "$PROJECT_ROOT/run_paper_trading_now.sh" << 'EOF'
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
EOF

# Make scripts executable
chmod +x "$PROJECT_ROOT/start_paper_trading.sh"
chmod +x "$PROJECT_ROOT/stop_paper_trading.sh" 
chmod +x "$PROJECT_ROOT/status_paper_trading.sh"
chmod +x "$PROJECT_ROOT/run_paper_trading_now.sh"

# Enable lingering for user systemd services
echo "🔧 Enabling user systemd lingering..."
if command -v loginctl &> /dev/null; then
    sudo loginctl enable-linger "$USER" || echo "⚠️ Could not enable lingering (requires sudo)"
else
    echo "⚠️ loginctl not available - you may need to manually enable lingering"
fi

echo ""
echo "✅ AUTOMATION SETUP COMPLETE"
echo "=============================="
echo ""
echo "📋 Next steps:"
echo "1. Review the generated systemd files in ~/.config/systemd/user/"
echo "2. Start automation: ./start_paper_trading.sh"
echo "3. Check status: ./status_paper_trading.sh" 
echo "4. Manual run: ./run_paper_trading_now.sh"
echo "5. Stop automation: ./stop_paper_trading.sh"
echo ""
echo "📅 SCHEDULE (all times in CT):"
echo "   08:00 - Preflight checks"
echo "   08:30 - Trading session starts"
echo "   15:10 - End of day operations"
echo ""
echo "🔍 MONITORING:"
echo "   Logs: journalctl --user -u paper-trading-*.service -f"
echo "   Reports: results/paper/reports/"
echo "   Status: ./status_paper_trading.sh"
echo ""
echo "🚨 EMERGENCY STOP:"
echo "   ./stop_paper_trading.sh"
echo "   OR: systemctl --user stop paper-trading-session.service"
