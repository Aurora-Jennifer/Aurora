#!/usr/bin/env python3
"""
Multi-Asset Universe Example

Example script showing how to run the multi-asset universe system.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def run_classical_universe():
    """Run classical models across the universe"""
    print("🚀 Running classical models across universe...")
    
    cmd = [
        "python", "scripts/run_universe.py",
        "--universe-cfg", "config/universe.yaml",
        "--grid-cfg", "config/grid.yaml", 
        "--out-dir", "results/classical_universe"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Classical universe run timed out")
        return False
    except Exception as e:
        print(f"❌ Classical universe run failed: {e}")
        return False


def run_dl_universe():
    """Run deep learning models across the universe"""
    print("🧠 Running deep learning models across universe...")
    
    cmd = [
        "python", "scripts/run_universe.py",
        "--universe-cfg", "config/universe.yaml",
        "--grid-cfg", "config/dl.yaml",
        "--out-dir", "results/dl_universe"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ DL universe run timed out")
        return False
    except Exception as e:
        print(f"❌ DL universe run failed: {e}")
        return False


def select_portfolio(leaderboard_path: str, k: int = 5):
    """Select top-K assets for portfolio"""
    print(f"🎯 Selecting top {k} assets from {leaderboard_path}...")
    
    cmd = [
        "python", "scripts/portfolio_from_leaderboard.py",
        "--leaderboard", leaderboard_path,
        "--k", str(k),
        "--create-config"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Portfolio selection failed: {e}")
        return False


def main():
    """Main example function"""
    print("🌟 Multi-Asset Universe Example")
    print("=" * 50)
    
    # Check if config files exist
    universe_cfg = Path("config/universe.yaml")
    grid_cfg = Path("config/grid.yaml")
    dl_cfg = Path("config/dl.yaml")
    
    if not universe_cfg.exists():
        print(f"❌ Universe config not found: {universe_cfg}")
        return 1
    
    if not grid_cfg.exists():
        print(f"❌ Grid config not found: {grid_cfg}")
        return 1
    
    if not dl_cfg.exists():
        print(f"❌ DL config not found: {dl_cfg}")
        return 1
    
    print("✅ All config files found")
    
    # Run classical universe
    print("\n1️⃣ Running classical models...")
    classical_success = run_classical_universe()
    
    if classical_success:
        print("✅ Classical universe completed")
        
        # Select portfolio from classical results
        classical_leaderboard = "results/classical_universe/leaderboard.csv"
        if Path(classical_leaderboard).exists():
            select_portfolio(classical_leaderboard, k=5)
    else:
        print("❌ Classical universe failed")
    
    # Run DL universe
    print("\n2️⃣ Running deep learning models...")
    dl_success = run_dl_universe()
    
    if dl_success:
        print("✅ DL universe completed")
        
        # Select portfolio from DL results
        dl_leaderboard = "results/dl_universe/leaderboard.csv"
        if Path(dl_leaderboard).exists():
            select_portfolio(dl_leaderboard, k=5)
    else:
        print("❌ DL universe failed")
    
    # Summary
    print("\n📊 Summary:")
    print(f"  Classical: {'✅ Success' if classical_success else '❌ Failed'}")
    print(f"  Deep Learning: {'✅ Success' if dl_success else '❌ Failed'}")
    
    if classical_success or dl_success:
        print("\n🎉 Multi-asset universe example completed!")
        print("📁 Check the results/ directory for outputs")
        return 0
    else:
        print("\n❌ All runs failed")
        return 1


if __name__ == "__main__":
    exit(main())
