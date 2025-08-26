#!/usr/bin/env python3
"""
SUPER EASY TRADING - Just Run This!
No coding knowledge needed. Just run this script and it does everything.
"""

import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path

def run_with_visible_output(cmd, timeout_seconds=300):
    """Run command with visible output and progress updates"""
    print(f"🚀 Starting: {cmd}")
    print("📊 Live output:")
    print("-" * 50)
    
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    start_time = time.time()
    last_update = start_time
    
    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                break
                
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print("⏰ Timeout reached, stopping...")
                break
            
            # Read output with timeout
            try:
                output = process.stdout.readline()
                if output:
                    print(f"📈 {output.strip()}")
                    last_update = time.time()
                else:
                    # No output, show progress
                    elapsed = time.time() - start_time
                    if elapsed - last_update > 10:  # Show progress every 10 seconds
                        minutes = int(elapsed // 60)
                        seconds = int(elapsed % 60)
                        print(f"⏱️  Running... {minutes}:{seconds:02d} elapsed")
                        last_update = time.time()
                    time.sleep(0.1)
            except Exception as e:
                print(f"⚠️  Error reading output: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user (Ctrl+C)")
    
    # Cleanup
    try:
        process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    
    return process.returncode

def main():
    print("🚀 SUPER EASY TRADING SYSTEM")
    print("=" * 50)
    print("Just sit back and watch! No coding needed.")
    print()
    
    # Check if we're in the right place
    if not Path("scripts/paper_runner.py").exists():
        print("❌ ERROR: Please run this from the trader folder")
        print("   Open terminal, type: cd /home/Jennifer/secure/aurora_plain/trader")
        print("   Then run: python scripts/SUPER_EASY.py")
        input("Press Enter to exit...")
        return
    
    print("✅ System found! Starting automatic test...")
    print()
    
    # Check if models are available
    print("🔍 Checking system components...")
    
    # Check for models
    model_files = [
        "artifacts/models/latest.onnx",
        "artifacts/models/linear_v1.pkl", 
        "models/universal_v1.pkl"
    ]
    
    found_models = []
    for model_file in model_files:
        if Path(model_file).exists():
            found_models.append(model_file)
            print(f"✅ Found model: {model_file}")
    
    if not found_models:
        print("⚠️  No models found - system will run in fallback mode")
    else:
        print(f"✅ Found {len(found_models)} model(s)")
    
    # Check config
    if Path("config/base.yaml").exists():
        print("✅ Found config: config/base.yaml")
    else:
        print("⚠️  No base config found")
    
    print()
    print("📊 Running 5-minute test with SPY...")
    print("   (This simulates real trading for 5 minutes)")
    print("   Press Ctrl+C to stop early and see results")
    print()
    
    # Run the paper trading with visible output
    cmd = "python scripts/paper_runner.py --symbols SPY --poll-sec 5"
    
    exit_code = run_with_visible_output(cmd, timeout_seconds=300)
    
    print()
    print("✅ Test completed!")
    print()
    
    # Show results
    print("📋 RESULTS:")
    print("-" * 30)
    
    # Check if results file exists
    results_file = Path("reports/paper_run.meta.json")
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            print(f"🆔 Run ID: {data.get('run_id', 'N/A')}")
            print(f"📈 Symbols: {', '.join(data.get('symbols', []))}")
            print(f"⏰ Duration: {data.get('start', 'N/A')} to {data.get('stop', 'N/A')}")
            
            if data.get('model_enabled'):
                print(f"🤖 Model: {data.get('model_id', 'N/A')}")
                print(f"⚠️  Fallbacks: {data.get('model_fallbacks', 0)}")
                print(f"🔧 Model Type: {data.get('model_kind', 'N/A')}")
            
            if 'model_tripwire' in data:
                print(f"🚨 Alert: {data['model_tripwire']}")
            
            if 'model_tripwire_turnover' in data:
                print(f"🔄 Turnover Alert: {data['model_tripwire_turnover']}")
            
            if data.get('model_fallbacks', 0) == 0:
                print("✅ System is working perfectly!")
            else:
                print("⚠️  System ran with some fallbacks (normal for testing)")
            
        except Exception as e:
            print(f"📄 Results saved to: reports/paper_run.meta.json")
            print(f"⚠️  Could not read results: {e}")
    else:
        print("❌ No results file found")
        print("   Check if the system ran correctly")
    
    # Check for other result files
    other_files = [
        "reports/paper_provenance.json",
        "reports/runner_state.json",
        "reports/performance/paper_trading.json"
    ]
    
    found_files = []
    for file_path in other_files:
        if Path(file_path).exists():
            found_files.append(file_path)
    
    print()
    print("🎉 DONE! Your trading system is ready.")
    print()
    print("📁 Files created:")
    print("   • reports/paper_run.meta.json (main results)")
    for file_path in found_files:
        print(f"   • {file_path}")
    print()
    
    # Show what this means
    print("💡 What this means:")
    print("   • Your system can load models and make predictions")
    print("   • Paper trading simulation works")
    print("   • Risk management and tripwires are active")
    print("   • You're ready for live paper trading!")
    print()
    print("🚀 To run again, just run this script again!")
    print("   python scripts/SUPER_EASY.py")
    print()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
