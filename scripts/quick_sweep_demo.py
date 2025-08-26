#!/usr/bin/env python3
"""
Quick demo of the professional config sweep system.

This demonstrates the "config lab" approach for systematic signal discovery
with discovery/confirmation timeline splits and statistical validation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Run a quick config sweep demonstration."""
    
    print("🧪 Aurora Professional Config Sweep Demo")
    print("=" * 50)
    
    # Check if we have the required profile
    profile_path = Path("config/experiment_profiles.yaml")
    if not profile_path.exists():
        print(f"❌ Profile file not found: {profile_path}")
        print("Please ensure config/experiment_profiles.yaml exists")
        return 1
    
    # Import after path setup
    try:
        from scripts.config_sweep import ConfigSweepRunner
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
        return 1
    
    print("✅ Dependencies loaded")
    
    # Create a minimal test profile
    test_profile_path = Path("config/test_quick_sweep.yaml")
    test_profile_content = """
name: "quick_sweep_demo"
hypothesis: "Demonstration of professional config sweep system"
feature_families: ["momentum_basic"]
tickers: ["SPY"]
discovery_start: "2023-01-01"
discovery_end: "2023-06-30"
confirmation_start: "2023-07-01"
confirmation_end: "2023-12-31"
trial_budget: 8
cost_bps: 5.0
min_ic_threshold: 0.01
min_sharpe_threshold: 0.2
random_seed: 42
parallel_jobs: 2
"""
    
    with open(test_profile_path, 'w') as f:
        f.write(test_profile_content)
    
    print(f"📝 Created test profile: {test_profile_path}")
    
    try:
        # Run the sweep
        print("\n🚀 Starting config sweep...")
        runner = ConfigSweepRunner(str(test_profile_path))
        
        # Run discovery only for demo
        print("\n🔍 Running discovery phase...")
        discovery_results = runner.run_discovery_phase()
        
        # Select candidates
        print("\n🎯 Selecting candidates...")
        candidates = runner.select_candidates(discovery_results)
        
        if candidates:
            print(f"\n✅ Demo complete! {len(candidates)} candidates ready for confirmation.")
            print("\nNext steps:")
            print("1. Review results in:", runner.exp_dir)
            print("2. Run confirmation phase with: --confirmation-only")
            print("3. Deploy best configs to paper trading")
        else:
            print("\n⚠️  No candidates passed filters.")
            print("This is normal for synthetic data - real features would perform better.")
        
        # Clean up
        test_profile_path.unlink()
        print(f"\n🧹 Cleaned up test profile")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
