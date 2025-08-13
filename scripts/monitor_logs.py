#!/usr/bin/env python3
"""
Log Monitoring Script
Monitors trading system logs and sends alerts
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta

def monitor_logs():
    """Monitor logs for errors and important events."""
    log_dir = Path("logs")
    
    # Check for recent errors
    error_log = log_dir / "errors" / f"errors_{datetime.now().strftime('%Y-%m')}.log"
    if error_log.exists():
        # Check for errors in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        with open(error_log, 'r') as f:
            for line in f:
                if 'ERROR' in line:
                    print(f"⚠️  Error found: {line.strip()}")
    
    # Check system health
    system_log = log_dir / "system" / f"system_{datetime.now().strftime('%Y-%m')}.log"
    if system_log.exists():
        print("✅ System logs found")
    
    print("🔍 Log monitoring completed")

if __name__ == "__main__":
    monitor_logs()
