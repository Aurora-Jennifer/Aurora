#!/bin/bash
# Simple Daily Paper Trading Script
# Run this manually each day or use the automated systemd timers

set -euo pipefail

cd "$(dirname "$0")"

# Auto-load environment if available
set -a
[ -f "$HOME/.config/paper-trading.env" ] && . "$HOME/.config/paper-trading.env"
set +a

# Set environment for paper trading
export IS_PAPER_TRADING=true
export TRADING_TIMEZONE=America/Chicago
export PYTHONPATH="$(pwd)"
export PYTHONHASHSEED=42

echo "📅 DAILY PAPER TRADING - $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

# Function to check current time and suggest appropriate mode
get_suggested_mode() {
    local hour=$(date '+%H')
    local minute=$(date '+%M')
    
    # Convert to Central Time (rough approximation)
    local ct_hour=$((hour - 6))  # Assuming CST (-6 UTC)
    if [ $ct_hour -lt 0 ]; then
        ct_hour=$((ct_hour + 24))
    fi
    
    if [ $ct_hour -lt 8 ]; then
        echo "preflight"
    elif [ $ct_hour -eq 8 ] && [ $minute -lt 30 ]; then
        echo "preflight"
    elif [ $ct_hour -eq 8 ] && [ $minute -ge 30 ]; then
        echo "trading"
    elif [ $ct_hour -gt 8 ] && [ $ct_hour -lt 15 ]; then
        echo "trading"
    elif [ $ct_hour -eq 15 ] && [ $minute -lt 30 ]; then
        echo "trading"
    else
        echo "eod"
    fi
}

# Parse command line arguments
MODE="${1:-auto}"

if [ "$MODE" = "auto" ]; then
    MODE=$(get_suggested_mode)
    echo "🤖 Auto-detected mode: $MODE (based on current time)"
fi

case "$MODE" in
    "preflight"|"pre")
        echo "🌅 Running pre-market preflight checks..."
        python3 ops/daily_paper_trading.py --mode preflight
        ;;
    
    "trading"|"trade"|"session")
        echo "📈 Running trading session..."
        python3 ops/daily_paper_trading.py --mode trading
        ;;
    
    "eod"|"end")
        echo "🌆 Running end-of-day operations..."
        python3 ops/daily_paper_trading.py --mode eod
        ;;
    
    "fetch"|"data-fetch")
        echo "📊 Fetching market data..."
        python3 tools/fetch_bars_alpaca.py --symbols-file data/universe/top300.txt \
            --start $(date -d "yesterday" +%Y-%m-%d) --end $(date -d "yesterday" +%Y-%m-%d) \
            --timeframe 1Day --feed iex --out data/latest/prices.parquet
        ;;
    
    "full"|"all")
        echo "🔄 Running full daily cycle..."
        python3 ops/daily_paper_trading.py --mode full
        ;;
    
    "status"|"check")
        echo "📊 Checking system status..."
        if [ -f "./status_paper_trading.sh" ]; then
            ./status_paper_trading.sh
        else
            echo "❌ Status script not found. Run setup first."
            exit 1
        fi
        ;;
    
    "setup")
        echo "🔧 Setting up automation..."
        echo "Choose automation type:"
        echo "  1) Systemd timers (recommended)"
        echo "  2) Cron jobs (universal)"
        read -p "Choice (1/2): " AUTOMATION_TYPE
        
        case "$AUTOMATION_TYPE" in
            "1"|"systemd"|"")
                ./ops/setup_paper_trading_automation.sh
                ;;
            "2"|"cron")
                ./ops/setup_cron_automation.sh
                ;;
            *)
                echo "❌ Invalid choice. Run 'setup' again."
                exit 1
                ;;
        esac
        ;;
    
    "start")
        echo "🚀 Starting automation..."
        if [ -f "./start_paper_trading.sh" ]; then
            ./start_paper_trading.sh
        elif [ -f "./start_cron_trading.sh" ]; then
            ./start_cron_trading.sh
        else
            echo "❌ No automation found. Run 'setup' first."
            exit 1
        fi
        ;;
    
    "stop")
        echo "🛑 Stopping automation..."
        if [ -f "./stop_paper_trading.sh" ]; then
            ./stop_paper_trading.sh
        elif [ -f "./stop_cron_trading.sh" ]; then
            ./stop_cron_trading.sh
        else
            echo "❌ No automation found to stop."
            exit 1
        fi
        ;;
    
    "tag")
        echo "🏷️ Tagging v1.0.0 release..."
        python3 ops/tag_v1_release.py
        ;;
    
    "help"|"-h"|"--help")
        echo "📖 DAILY PAPER TRADING USAGE"
        echo "============================="
        echo ""
        echo "Modes:"
        echo "  auto         - Auto-detect mode based on current time (default)"
        echo "  preflight    - Run pre-market checks (08:00 CT)"
        echo "  trading      - Run trading session (08:30-15:00 CT)"
        echo "  eod          - Run end-of-day operations (15:10 CT)"
        echo "  full         - Run complete daily cycle"
        echo ""
        echo "Management:"
        echo "  setup        - Set up automation (systemd timers or cron)"
        echo "  start        - Start automation"
        echo "  stop         - Stop automation"
        echo "  status       - Check system status"
        echo "  tag          - Tag v1.0.0 release"
        echo ""
        echo "Examples:"
        echo "  ./daily_paper_trading.sh           # Auto-detect and run appropriate mode"
        echo "  ./daily_paper_trading.sh preflight # Run preflight checks"
        echo "  ./daily_paper_trading.sh full      # Run full daily cycle"
        echo "  ./daily_paper_trading.sh setup     # Set up automation"
        echo "  ./daily_paper_trading.sh start     # Start automated timers"
        echo ""
        echo "Schedule (Central Time):"
        echo "  08:00 CT - Preflight checks"
        echo "  08:30 CT - Trading session starts"
        echo "  15:10 CT - End-of-day operations"
        ;;
    
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Run './daily_paper_trading.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "✅ Daily paper trading operation completed"
echo "📊 Check results/paper/ for reports and logs"
