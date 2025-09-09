#!/usr/bin/env python3
"""
Easy Trading Script - One Button Operation
For non-technical users who just want to test the system
"""

import os
import signal
import subprocess
import sys
from pathlib import Path


def run_with_timeout(cmd, timeout_seconds):
    """Run a command with a timeout"""
    try:
        print(f"   Running: {cmd}")
        print(f"   Will stop after {timeout_seconds} seconds...")
        
        # Start the process
        process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        
        try:
            # Wait for the process to complete or timeout
            process.wait(timeout=timeout_seconds)
            return process.returncode
        except subprocess.TimeoutExpired:
            # Kill the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=5)  # Give it 5 seconds to terminate gracefully
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)  # Force kill
            print(f"   Stopped after {timeout_seconds} seconds")
            return 0
    except Exception as e:
        print(f"   Error: {e}")
        return 1

def main():
    print("🎯 EASY TRADING SYSTEM")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("scripts/paper_runner.py").exists():
        print("❌ ERROR: Please run this from the trader directory")
        print("   cd /home/Jennifer/secure/aurora_plain/trader")
        return 1
    
    # Simple menu
    print("\nChoose what you want to do:")
    print("1. Quick test (SPY, 5 minutes)")
    print("2. Longer test (SPY+QQQ, 10 minutes)")
    print("3. Crypto test (BTC+ETH, 30 minutes)")
    print("4. Check if system is working")
    print("5. Exit")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    
    if choice == "1":
        print("\n🚀 Starting quick test...")
        print("   Testing SPY for 5 minutes...")
        cmd = "python scripts/paper_runner.py --symbols SPY --poll-sec 5"
        run_with_timeout(cmd, 5 * 60)  # 5 minutes
        
    elif choice == "2":
        print("\n🚀 Starting longer test...")
        print("   Testing SPY and QQQ for 10 minutes...")
        cmd = "python scripts/paper_runner.py --symbols SPY,QQQ --poll-sec 5"
        run_with_timeout(cmd, 10 * 60)  # 10 minutes
        
    elif choice == "3":
        print("\n🚀 Starting crypto test...")
        print("   Testing Bitcoin and Ethereum for 30 minutes...")
        cmd = "python scripts/paper_runner.py --symbols BTCUSDT,ETHUSDT --poll-sec 5"
        run_with_timeout(cmd, 30 * 60)  # 30 minutes
        
    elif choice == "4":
        print("\n🔍 Checking system...")
        os.system("make smoke")
        
    elif choice == "5":
        print("\n👋 Goodbye!")
        return 0
        
    else:
        print("\n❌ Invalid choice. Please enter 1-5.")
        return 1
    
    print("\n✅ Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
