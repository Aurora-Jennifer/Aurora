#!/usr/bin/env python3
"""
IBKR Gateway Connection Diagnostic Tool
Helps troubleshoot connection issues with IBKR Gateway
"""

import socket
import subprocess
import sys
from datetime import datetime


def check_port(host, port):
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error checking port {port}: {e}")
        return False


def check_processes():
    """Check for IBKR-related processes."""
    try:
        # Check for IB Gateway or TWS processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = result.stdout.split("\n")

        ibkr_processes = []
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["ibgateway", "tws", "ibkr", "interactive"]
            ):
                ibkr_processes.append(line.strip())

        return ibkr_processes
    except Exception as e:
        print(f"Error checking processes: {e}")
        return []


def test_connection(host, port):
    """Test connection to IBKR Gateway."""
    print(f"\n🔍 Testing connection to {host}:{port}")

    # Check if port is open
    if check_port(host, port):
        print(f"✅ Port {port} is open and accepting connections")
        return True
    else:
        print(f"❌ Port {port} is not open or not accepting connections")
        return False


def main():
    """Main diagnostic function."""
    print("=" * 60)
    print("IBKR Gateway Connection Diagnostic")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")

    # Check common IBKR ports
    ports_to_check = [
        (7497, "Paper Trading"),
        (7496, "Live Trading"),
        (4001, "Paper Trading (Alternative)"),
        (4002, "Live Trading (Alternative)"),
    ]

    print("\n📋 Checking IBKR Gateway Status...")

    # Check for IBKR processes
    processes = check_processes()
    if processes:
        print("✅ Found IBKR-related processes:")
        for process in processes:
            print(f"   {process}")
    else:
        print("❌ No IBKR-related processes found")
        print("   Make sure IBKR Gateway is running")

    # Test connections
    print("\n🔌 Testing Port Connections...")
    open_ports = []

    for port, description in ports_to_check:
        if test_connection("127.0.0.1", port):
            open_ports.append((port, description))

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if open_ports:
        print("✅ Found open IBKR ports:")
        for port, description in open_ports:
            print(f"   Port {port}: {description}")

        print("\n🎯 RECOMMENDATIONS:")
        print("1. Update your configuration to use one of the open ports")
        print("2. If using port 7497 (paper trading), your config is correct")
        print("3. If using port 7496 (live trading), update your config")

        # Show configuration example
        if any(port == 7497 for port, _ in open_ports):
            print("\n📝 Example configuration for paper trading:")
            print(
                """
{
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": true,
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 1
  }
}
            """
            )

        if any(port == 7496 for port, _ in open_ports):
            print("\n📝 Example configuration for live trading:")
            print(
                """
{
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": false,
    "host": "127.0.0.1",
    "port": 7496,
    "client_id": 1
  }
}
            """
            )

    else:
        print("❌ No IBKR ports are open")
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Make sure IBKR Gateway is running")
        print("2. Check IBKR Gateway configuration:")
        print("   - Go to Configure > API > Settings")
        print("   - Enable 'Enable ActiveX and Socket Clients'")
        print("   - Set Socket port to 7497 (paper) or 7496 (live)")
        print("   - Set 'Read-Only API' to 'No'")
        print("3. Restart IBKR Gateway after configuration changes")
        print("4. Check firewall settings")
        print("5. Verify you're logged into the correct account")

    print("\n📚 Additional Resources:")
    print("- IBKR Setup Guide: IBKR_SETUP_GUIDE.md")
    print("- IBKR API Documentation: https://interactivebrokers.github.io/tws-api/")
    print("- IBKR Community: https://community.interactivebrokers.com/")

    return len(open_ports) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
