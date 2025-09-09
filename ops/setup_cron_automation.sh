#!/bin/bash
# Cron Job Setup for Paper Trading Automation
# Alternative to systemd timers for broader compatibility

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USER="$(whoami)"

echo "⏰ SETTING UP CRON JOB AUTOMATION"
echo "================================="
echo "Project root: $PROJECT_ROOT"
echo "User: $USER"

# Create wrapper scripts for cron (cron has minimal environment)
echo "📝 Creating cron wrapper scripts..."

# Preflight wrapper
cat > "$PROJECT_ROOT/ops/cron_preflight.sh" << EOF
#!/bin/bash
# Cron wrapper for preflight checks

# Set working directory
cd "$PROJECT_ROOT"

# Set environment (cron has minimal env)
export PATH="/usr/local/bin:/usr/bin:/bin"
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONHASHSEED=42
export PYTHONIOENCODING=utf-8

# Ensure log directory exists
mkdir -p logs

# Run with output redirection
exec python3 ops/daily_paper_trading.py --mode preflight >> logs/cron_preflight.log 2>&1
EOF

# Trading session wrapper
cat > "$PROJECT_ROOT/ops/cron_trading.sh" << EOF
#!/bin/bash
# Cron wrapper for trading session

# Set working directory
cd "$PROJECT_ROOT"

# Set environment
export PATH="/usr/local/bin:/usr/bin:/bin"
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONHASHSEED=42
export PYTHONIOENCODING=utf-8

# Ensure log directory exists
mkdir -p logs

# Run with output redirection
exec python3 ops/daily_paper_trading.py --mode trading >> logs/cron_trading.log 2>&1
EOF

# EOD wrapper
cat > "$PROJECT_ROOT/ops/cron_eod.sh" << EOF
#!/bin/bash
# Cron wrapper for end-of-day operations

# Set working directory
cd "$PROJECT_ROOT"

# Set environment
export PATH="/usr/local/bin:/usr/bin:/bin"
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONHASHSEED=42
export PYTHONIOENCODING=utf-8

# Ensure log directory exists
mkdir -p logs

# Run with output redirection
exec python3 ops/daily_paper_trading.py --mode eod >> logs/cron_eod.log 2>&1
EOF

# Make wrapper scripts executable
chmod +x "$PROJECT_ROOT/ops/cron_preflight.sh"
chmod +x "$PROJECT_ROOT/ops/cron_trading.sh"
chmod +x "$PROJECT_ROOT/ops/cron_eod.sh"

echo "✅ Cron wrapper scripts created"

# Generate crontab entries
echo ""
echo "📋 CRONTAB ENTRIES TO ADD:"
echo "=========================="
echo ""
echo "# Paper Trading Automation (Central Time)"
echo "# 08:00 CT = 14:00 UTC (preflight)"
echo "0 14 * * 1-5 $PROJECT_ROOT/ops/cron_preflight.sh"
echo ""
echo "# 08:30 CT = 14:30 UTC (trading session)"  
echo "30 14 * * 1-5 $PROJECT_ROOT/ops/cron_trading.sh"
echo ""
echo "# 15:10 CT = 21:10 UTC (end of day)"
echo "10 21 * * 1-5 $PROJECT_ROOT/ops/cron_eod.sh"
echo ""

# Create crontab file
CRON_FILE="/tmp/paper_trading_crontab.txt"
cat > "$CRON_FILE" << EOF
# Paper Trading Automation (Central Time)
# Generated on $(date)

# 08:00 CT = 14:00 UTC (preflight checks)
0 14 * * 1-5 $PROJECT_ROOT/ops/cron_preflight.sh

# 08:30 CT = 14:30 UTC (trading session)
30 14 * * 1-5 $PROJECT_ROOT/ops/cron_trading.sh

# 15:10 CT = 21:10 UTC (end of day operations)
10 21 * * 1-5 $PROJECT_ROOT/ops/cron_eod.sh
EOF

echo "💾 Cron entries saved to: $CRON_FILE"
echo ""

# Offer to install crontab
echo "🤖 INSTALLATION OPTIONS:"
echo "========================"
echo ""
echo "Option 1 - Automatic install:"
echo "   crontab $CRON_FILE"
echo ""
echo "Option 2 - Manual install:"
echo "   crontab -e"
echo "   # Then copy/paste the entries above"
echo ""
echo "Option 3 - Merge with existing crontab:"
echo "   crontab -l > /tmp/current_cron.txt 2>/dev/null || true"
echo "   cat $CRON_FILE >> /tmp/current_cron.txt"
echo "   crontab /tmp/current_cron.txt"
echo ""

read -p "Install crontab automatically? (y/N): " AUTO_INSTALL

if [[ "$AUTO_INSTALL" =~ ^[Yy]$ ]]; then
    echo "🚀 Installing crontab..."
    
    # Backup existing crontab
    BACKUP_FILE="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
    if crontab -l > "$BACKUP_FILE" 2>/dev/null; then
        echo "📄 Existing crontab backed up to: $BACKUP_FILE"
        
        # Merge with existing
        cat "$CRON_FILE" >> "$BACKUP_FILE"
        crontab "$BACKUP_FILE"
        echo "✅ Crontab updated (merged with existing)"
    else
        # No existing crontab
        crontab "$CRON_FILE"
        echo "✅ Crontab installed (new)"
    fi
else
    echo "ℹ️ Manual installation required"
    echo "   Run: crontab $CRON_FILE"
fi

# Create management scripts for cron
cat > "$PROJECT_ROOT/start_cron_trading.sh" << 'EOF'
#!/bin/bash
# Start cron-based paper trading

echo "⏰ STARTING CRON PAPER TRADING"
echo "=============================="

# Check if cron entries are installed
if crontab -l 2>/dev/null | grep -q "paper_trading"; then
    echo "✅ Cron entries found"
    echo ""
    echo "📋 Active cron schedule:"
    crontab -l | grep paper_trading
    echo ""
    echo "🔍 Monitor logs:"
    echo "   tail -f logs/cron_*.log"
else
    echo "❌ No paper trading cron entries found"
    echo "   Run: ./ops/setup_cron_automation.sh"
    exit 1
fi
EOF

cat > "$PROJECT_ROOT/stop_cron_trading.sh" << 'EOF'
#!/bin/bash
# Stop cron-based paper trading

echo "🛑 STOPPING CRON PAPER TRADING"
echo "=============================="

# Backup current crontab
BACKUP_FILE="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
if crontab -l > "$BACKUP_FILE" 2>/dev/null; then
    echo "📄 Current crontab backed up to: $BACKUP_FILE"
    
    # Remove paper trading entries
    crontab -l | grep -v "paper_trading\|cron_preflight\|cron_trading\|cron_eod" > /tmp/new_cron.txt
    crontab /tmp/new_cron.txt
    
    echo "✅ Paper trading entries removed from crontab"
    echo "📄 Backup available at: $BACKUP_FILE"
else
    echo "ℹ️ No existing crontab found"
fi
EOF

cat > "$PROJECT_ROOT/status_cron_trading.sh" << 'EOF'
#!/bin/bash
# Check cron-based paper trading status

echo "📊 CRON PAPER TRADING STATUS"
echo "============================"

echo ""
echo "⏰ CRON ENTRIES:"
if crontab -l 2>/dev/null | grep -q "paper_trading\|cron_"; then
    crontab -l | grep -E "paper_trading|cron_"
else
    echo "   No paper trading cron entries found"
fi

echo ""
echo "📄 LOG FILES:"
if [ -d "logs" ]; then
    for log in logs/cron_*.log; do
        if [ -f "$log" ]; then
            echo "   $(ls -la "$log")"
        fi
    done
else
    echo "   No log directory found"
fi

echo ""
echo "📋 RECENT LOG ACTIVITY (last 20 lines):"
for log in logs/cron_*.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "--- $log ---"
        tail -20 "$log" 2>/dev/null || echo "   (empty or unreadable)"
    fi
done

echo ""
echo "📁 RECENT REPORTS:"
if [ -d "results/paper/reports" ]; then
    ls -la results/paper/reports/ | tail -5
else
    echo "   No reports directory found"
fi
EOF

# Make management scripts executable
chmod +x "$PROJECT_ROOT/start_cron_trading.sh"
chmod +x "$PROJECT_ROOT/stop_cron_trading.sh"
chmod +x "$PROJECT_ROOT/status_cron_trading.sh"

echo ""
echo "✅ CRON AUTOMATION SETUP COMPLETE"
echo "=================================="
echo ""
echo "📋 Created files:"
echo "   Wrapper scripts: ops/cron_*.sh"
echo "   Crontab file: $CRON_FILE"
echo "   Management: start_cron_trading.sh, stop_cron_trading.sh, status_cron_trading.sh"
echo ""
echo "⏰ Schedule (Central Time):"
echo "   08:00 CT - Preflight checks"
echo "   08:30 CT - Trading session"
echo "   15:10 CT - End of day operations"
echo ""
echo "📝 Log files will be created in: logs/cron_*.log"
echo ""
echo "🔧 Management commands:"
echo "   ./start_cron_trading.sh   - Check status"
echo "   ./stop_cron_trading.sh    - Remove cron entries"
echo "   ./status_cron_trading.sh  - View logs and status"
