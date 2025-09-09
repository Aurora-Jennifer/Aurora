#!/usr/bin/env python3
"""
Professional experiment runner for systematic signal discovery.

This script makes it easy to run large-scale configuration sweeps across
different tickers, features, and model parameters with statistical validation.

Usage Examples:
    # Quick validation experiment
    python scripts/experiment_runner.py quick_validation
    
    # Momentum discovery across multiple assets
    python scripts/experiment_runner.py momentum_discovery
    
    # Conservative validation with strict thresholds
    python scripts/experiment_runner.py conservative_validation
    
    # Custom experiment with discovery only
    python scripts/experiment_runner.py momentum_discovery --discovery-only
    
    # Dry run to see config generation
    python scripts/experiment_runner.py momentum_discovery --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_available_experiments():
    """List all available experiment profiles."""
    import yaml
    
    profile_path = Path("config/experiment_profiles.yaml")
    if not profile_path.exists():
        print("❌ No experiment profiles found at config/experiment_profiles.yaml")
        return []
    
    with open(profile_path) as f:
        profiles = yaml.safe_load(f)
    
    print("📋 Available Experiment Profiles:")
    print("=" * 50)
    
    for name, profile in profiles.items():
        hypothesis = profile.get('hypothesis', 'No hypothesis provided')
        budget = profile.get('trial_budget', 'Unknown')
        tickers = profile.get('tickers', [])
        
        print(f"\n🧪 {name}")
        print(f"   Hypothesis: {hypothesis}")
        print(f"   Trial Budget: {budget}")
        print(f"   Tickers: {', '.join(tickers)}")
        print(f"   Timeline: {profile.get('discovery_start', '?')} to {profile.get('confirmation_end', '?')}")
    
    return list(profiles.keys())


def run_experiment(experiment_name: str, args: argparse.Namespace):
    """Run a specific experiment."""
    
    # Import config sweep runner
    try:
        from scripts.experiments.config_sweep import ConfigSweepRunner
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
        return 1
    
    # Build profile path
    profile_file = "config/experiment_profiles.yaml"
    profile_path = Path(profile_file)
    
    if not profile_path.exists():
        print(f"❌ Profile file not found: {profile_path}")
        return 1
    
    # Create temporary single-experiment profile
    import yaml
    with open(profile_path) as f:
        all_profiles = yaml.safe_load(f)
    
    if experiment_name not in all_profiles:
        print(f"❌ Experiment '{experiment_name}' not found")
        print(f"Available experiments: {list(all_profiles.keys())}")
        return 1
    
    # Extract single experiment profile
    single_profile = all_profiles[experiment_name]
    temp_profile_path = Path(f"config/temp_{experiment_name}.yaml")
    
    with open(temp_profile_path, 'w') as f:
        yaml.dump(single_profile, f, indent=2)
    
    try:
        # Run the experiment
        runner = ConfigSweepRunner(str(temp_profile_path))
        
        if args.dry_run:
            configs = runner.generate_configs()
            print(f"\n🔧 Would generate {len(configs)} configurations")
            
            # Show first few configs as examples
            for i, config in enumerate(configs[:3]):
                print(f"\nExample Config {i+1}:")
                print(f"  Ticker: {config['ticker']}")
                print(f"  Features: {config['feature_set']}")
                print(f"  Model: {config['model_config']['model_type']}")
                if 'alpha' in config['model_config']:
                    print(f"  Alpha: {config['model_config']['alpha']}")
                if 'n_estimators' in config['model_config']:
                    print(f"  N-Estimators: {config['model_config']['n_estimators']}")
            
            if len(configs) > 3:
                print(f"\n... and {len(configs) - 3} more configurations")
            
            return 0
        
        if args.discovery_only:
            # Discovery phase only
            discovery_results = runner.run_discovery_phase()
            candidates = runner.select_candidates(discovery_results)
            
            print("\n🔍 Discovery complete!")
            print(f"Configs tested: {len(discovery_results)}")
            print(f"Candidates for confirmation: {len(candidates)}")
            
            if candidates:
                print("\nTo run confirmation phase:")
                print(f"python scripts/config_sweep.py {temp_profile_path}")
            
            return 0
        
        # Full experiment (discovery + confirmation)
        report = runner.run_full_experiment()
        
        # Success based on promoted configs
        promoted = report['confirmation_phase']['promoted_configs']
        if promoted:
            print(f"\n🚀 SUCCESS: {len(promoted)} configurations promoted!")
            print("Ready for paper trading deployment")
            return 0
        print("\n⚠️  No configurations passed confirmation")
        print("Consider adjusting thresholds or features")
        return 1
            
    finally:
        # Clean up temporary profile
        if temp_profile_path.exists():
            temp_profile_path.unlink()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Professional experiment runner for systematic signal discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "experiment", 
        nargs='?',
        help="Experiment name (use 'list' to see available experiments)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Generate configs but don't run experiments"
    )
    parser.add_argument(
        "--discovery-only", 
        action="store_true",
        help="Run discovery phase only"
    )
    
    args = parser.parse_args()
    
    if not args.experiment or args.experiment == "list":
        available = list_available_experiments()
        if available:
            print("\nUsage: python scripts/experiment_runner.py <experiment_name>")
            print(f"Example: python scripts/experiment_runner.py {available[0]}")
        return 0
    
    return run_experiment(args.experiment, args)


if __name__ == "__main__":
    sys.exit(main())
