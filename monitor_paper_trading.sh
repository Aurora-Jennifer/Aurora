#!/bin/bash
# Paper Trading Monitoring Commands

echo "📊 PAPER TRADING SYSTEM STATUS"
echo "=============================="
echo ""

echo "⏰ SYSTEMD TIMERS:"
systemctl --user list-timers paper-* --no-pager

echo ""
echo "📈 SERVICE STATUS:"
for service in paper-preflight paper-status paper-eod paper-data-fetch; do
    echo "[$service]"
    systemctl --user is-active $service.service || echo "  Inactive (normal for oneshot)"
done

echo ""
echo "📋 RECENT LOGS:"
echo "[Preflight]"
tail -n 5 logs/systemd_preflight.log 2>/dev/null || echo "  No preflight logs yet"
echo "[Status]"  
tail -n 5 logs/systemd_status.log 2>/dev/null || echo "  No status logs yet"
echo "[EOD]"
tail -n 5 logs/systemd_eod.log 2>/dev/null || echo "  No EOD logs yet"

echo ""
echo "🚨 EMERGENCY COMMANDS:"
echo "  Kill-switch: touch kill.flag"
echo "  Live logs:   journalctl --user -u paper-* -f"
echo "  Manual test: systemctl --user start paper-preflight.service"
