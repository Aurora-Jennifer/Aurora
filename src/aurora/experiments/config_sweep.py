#!/usr/bin/env python3
"""
Professional-grade configuration sweep runner with statistical validation.

This implements the "config lab" approach for systematic signal discovery:
- Timeline splits (discovery vs confirmation holdout)
- IC significance testing with HAC standard errors
- Multiple testing controls (deflated Sharpe)
- Block bootstrap validation
- Full experiment provenance tracking
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stats.ic_validator import ICResult, ICValidator
from scripts.training.train import run_training_experiment


@dataclass
class ExperimentProfile:
    """Experiment configuration with search budget and guardrails."""
    name: str
    hypothesis: str
    feature_families: list[str]
    tickers: list[str]
    discovery_start: str  # ISO date
    discovery_end: str    # ISO date
    confirmation_start: str  # ISO date (must be > discovery_end)
    confirmation_end: str    # ISO date
    trial_budget: int = 200
    cost_bps: float = 5.0
    min_ic_threshold: float = 0.02
    min_sharpe_threshold: float = 0.5
    random_seed: int = 42
    parallel_jobs: int = 4


@dataclass
class ConfigResult:
    """Results from a single configuration trial."""
    config_id: str
    ticker: str
    feature_set: str
    model_type: str
    ic_stats: ICResult
    sharpe: float
    max_drawdown: float
    annual_return: float
    total_trades: int
    metadata: dict[str, Any]
    timestamp: str


class ConfigSweepRunner:
    """Professional configuration sweep runner with statistical validation."""
    
    def __init__(self, profile_path: str):
        """Initialize with experiment profile."""
        self.profile = self._load_profile(profile_path)
        self.ic_validator = ICValidator()
        self.results: list[ConfigResult] = []
        self.confirmation_locked = False
        
        # Create experiment directory
        self.exp_dir = Path(f"reports/experiments/{self.profile.name}_{int(time.time())}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧪 Experiment: {self.profile.name}")
        print(f"📁 Results: {self.exp_dir}")
        print(f"🎯 Hypothesis: {self.profile.hypothesis}")
        print(f"💰 Trial budget: {self.profile.trial_budget}")
    
    def _load_profile(self, profile_path: str) -> ExperimentProfile:
        """Load and validate experiment profile."""
        with open(profile_path) as f:
            data = yaml.safe_load(f)
        
        profile = ExperimentProfile(**data)
        
        # Validate timeline
        disc_start = pd.to_datetime(profile.discovery_start)
        disc_end = pd.to_datetime(profile.discovery_end)
        conf_start = pd.to_datetime(profile.confirmation_start)
        conf_end = pd.to_datetime(profile.confirmation_end)
        
        if disc_start >= disc_end:
            raise ValueError("Discovery start must be before discovery end")
        if disc_end >= conf_start:
            raise ValueError("Discovery end must be before confirmation start")
        if conf_start >= conf_end:
            raise ValueError("Confirmation start must be before confirmation end")
        
        print(f"📅 Discovery: {profile.discovery_start} to {profile.discovery_end}")
        print(f"🔒 Confirmation: {profile.confirmation_start} to {profile.confirmation_end}")
        
        return profile
    
    def generate_configs(self) -> list[dict[str, Any]]:
        """Generate configuration grid for systematic testing."""
        configs = []
        config_id = 0
        
        # Feature engineering variations
        feature_sets = {
            "momentum_basic": ["momentum_3d", "momentum_5d", "momentum_10d"],
            "momentum_extended": ["momentum_3d", "momentum_5d", "momentum_10d", "momentum_20d", "momentum_strength"],
            "volatility_focus": ["vol_5", "vol_10", "vol_20", "rsi_14"],
            "combined_signals": ["momentum_5d", "momentum_20d", "vol_10", "rsi_14", "sma_ratio"]
        }
        
        # Model variations
        model_configs = {
            "ridge_conservative": {"model_type": "ridge", "alpha": 1.0},
            "ridge_aggressive": {"model_type": "ridge", "alpha": 0.1},
            "xgboost_light": {"model_type": "xgboost", "n_estimators": 50, "max_depth": 3},
            "xgboost_deep": {"model_type": "xgboost", "n_estimators": 100, "max_depth": 6}
        }
        
        # Generate combinations
        for ticker in self.profile.tickers:
            for feat_name, features in feature_sets.items():
                if feat_name not in self.profile.feature_families:
                    continue
                    
                for model_name, model_params in model_configs.items():
                    config = {
                        "config_id": f"cfg_{config_id:04d}",
                        "ticker": ticker,
                        "feature_set": feat_name,
                        "features": features,
                        "model_config": model_params,
                        "cost_bps": self.profile.cost_bps,
                        "random_seed": self.profile.random_seed + config_id
                    }
                    configs.append(config)
                    config_id += 1
                    
                    if config_id >= self.profile.trial_budget:
                        print(f"⚠️  Hit trial budget ({self.profile.trial_budget}), truncating search")
                        return configs
        
        print(f"🔧 Generated {len(configs)} configurations")
        return configs
    
    def run_discovery_phase(self) -> list[ConfigResult]:
        """Run discovery phase with timeline isolation."""
        print("\n🔍 DISCOVERY PHASE")
        print(f"Timeline: {self.profile.discovery_start} to {self.profile.discovery_end}")
        
        configs = self.generate_configs()
        discovery_results = []
        
        # Run configurations in parallel
        with ProcessPoolExecutor(max_workers=self.profile.parallel_jobs) as executor:
            # Submit all jobs
            future_to_config = {}
            for config in configs:
                future = executor.submit(self._run_single_config, config, phase="discovery")
                future_to_config[future] = config
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_config)):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result:
                        discovery_results.append(result)
                        print(f"✅ [{i+1:3d}/{len(configs)}] {config['config_id']} | IC: {result.ic_stats.mean_ic:.4f} | Sharpe: {result.sharpe:.2f}")
                except Exception as e:
                    print(f"❌ [{i+1:3d}/{len(configs)}] {config['config_id']} failed: {e}")
        
        # Save discovery results
        self._save_results(discovery_results, "discovery_results.json")
        
        # Statistical summary
        if discovery_results:
            ics = [r.ic_stats.mean_ic for r in discovery_results]
            sharpes = [r.sharpe for r in discovery_results]
            print("\n📊 DISCOVERY SUMMARY")
            print(f"Configs tested: {len(discovery_results)}")
            print(f"IC stats: μ={np.mean(ics):.4f}, σ={np.std(ics):.4f}, range=[{np.min(ics):.4f}, {np.max(ics):.4f}]")
            print(f"Sharpe stats: μ={np.mean(sharpes):.3f}, σ={np.std(sharpes):.3f}, range=[{np.min(sharpes):.3f}, {np.max(sharpes):.3f}]")
        
        return discovery_results
    
    def select_candidates(self, discovery_results: list[ConfigResult]) -> list[ConfigResult]:
        """Select candidate configurations for confirmation testing."""
        if not discovery_results:
            print("❌ No discovery results to select from")
            return []
        
        # Filter by thresholds
        candidates = []
        for result in discovery_results:
            # IC significance check
            if result.ic_stats.t_stat < 2.0:  # Not statistically significant
                continue
            
            # Economic significance checks
            if result.ic_stats.mean_ic < self.profile.min_ic_threshold:
                continue
            if result.sharpe < self.profile.min_sharpe_threshold:
                continue
            if result.max_drawdown > 0.3:  # Max 30% drawdown
                continue
            
            candidates.append(result)
        
        # Sort by deflated Sharpe (simple approximation)
        # Deflated Sharpe ≈ Sharpe * sqrt(1 - log(num_trials) / num_observations)
        num_trials = len(discovery_results)
        deflation_factor = np.sqrt(1 - np.log(num_trials) / 252)  # Assume daily data
        
        for candidate in candidates:
            candidate.deflated_sharpe = candidate.sharpe * deflation_factor
        
        candidates.sort(key=lambda x: x.deflated_sharpe, reverse=True)
        
        # Take top 5 or 10% of budget, whichever is smaller
        max_candidates = min(5, max(1, int(0.1 * self.profile.trial_budget)))
        top_candidates = candidates[:max_candidates]
        
        print("\n🎯 CANDIDATE SELECTION")
        print(f"Discovery configs: {len(discovery_results)}")
        print(f"Passed filters: {len(candidates)}")
        print(f"Selected for confirmation: {len(top_candidates)}")
        
        for i, candidate in enumerate(top_candidates):
            print(f"  {i+1}. {candidate.config_id} | IC: {candidate.ic_stats.mean_ic:.4f} (t={candidate.ic_stats.t_stat:.2f}) | Sharpe: {candidate.sharpe:.2f} | Deflated: {candidate.deflated_sharpe:.2f}")
        
        return top_candidates
    
    def run_confirmation_phase(self, candidates: list[ConfigResult]) -> list[ConfigResult]:
        """Run confirmation phase on quarantined holdout data."""
        if not candidates:
            print("❌ No candidates for confirmation")
            return []
        
        print("\n🔒 CONFIRMATION PHASE")
        print(f"Timeline: {self.profile.confirmation_start} to {self.profile.confirmation_end}")
        print("⚠️  HOLDOUT DATA - ONE SHOT ONLY")
        
        # Lock confirmation to prevent multiple runs
        if self.confirmation_locked:
            raise RuntimeError("Confirmation phase already run! Must start new experiment.")
        self.confirmation_locked = True
        
        confirmation_results = []
        
        # Run each candidate on holdout data
        for i, candidate in enumerate(candidates):
            print(f"\n🧪 Testing candidate {i+1}/{len(candidates)}: {candidate.config_id}")
            
            # Reconstruct config from candidate metadata
            config = candidate.metadata.get("config", {})
            config["config_id"] = f"{candidate.config_id}_confirmation"
            
            try:
                result = self._run_single_config(config, phase="confirmation")
                if result:
                    confirmation_results.append(result)
                    print(f"✅ Confirmation IC: {result.ic_stats.mean_ic:.4f} (t={result.ic_stats.t_stat:.2f}) | Sharpe: {result.sharpe:.2f}")
                else:
                    print("❌ Confirmation failed")
            except Exception as e:
                print(f"❌ Confirmation error: {e}")
        
        # Save confirmation results
        self._save_results(confirmation_results, "confirmation_results.json")
        
        return confirmation_results
    
    def _run_single_config(self, config: dict[str, Any], phase: str) -> ConfigResult | None:
        """Run a single configuration trial."""
        try:
            # Set timeline based on phase
            if phase == "discovery":
                start_date = self.profile.discovery_start
                end_date = self.profile.discovery_end
            else:  # confirmation
                start_date = self.profile.confirmation_start
                end_date = self.profile.confirmation_end
            
            # Build training arguments
            train_args = {
                "model_type": config["model_config"]["model_type"],
                "symbol": config["ticker"],
                "cost_bps": config["cost_bps"],
                "start_date": start_date,
                "end_date": end_date,
                "features": config["features"],
                "random_seed": config["random_seed"],
                **config["model_config"]
            }
            
            # Run training experiment
            experiment_result = run_training_experiment(train_args)
            
            if not experiment_result or "metrics" not in experiment_result:
                return None
            
            metrics = experiment_result["metrics"]
            
            # Compute IC statistics
            if "predictions" in experiment_result and "returns" in experiment_result:
                predictions = experiment_result["predictions"]
                returns = experiment_result["returns"]
                ic_result = self.ic_validator.compute_ic_stats(predictions, returns)
            else:
                # Fallback if no prediction series available
                ic_result = ICResult(
                    mean_ic=metrics.get("ic", 0.0),
                    ic_std=0.0,
                    t_stat=0.0,
                    p_value=1.0,
                    confidence_interval=(0.0, 0.0),
                    hit_rate=0.5,
                    num_observations=0
                )
            
            # Build result
            result = ConfigResult(
                config_id=config["config_id"],
                ticker=config["ticker"],
                feature_set=config["feature_set"],
                model_type=config["model_config"]["model_type"],
                ic_stats=ic_result,
                sharpe=metrics.get("sharpe", 0.0),
                max_drawdown=metrics.get("max_drawdown", 1.0),
                annual_return=metrics.get("annual_return", 0.0),
                total_trades=metrics.get("total_trades", 0),
                metadata={
                    "config": config,
                    "experiment_id": experiment_result.get("exp_id"),
                    "phase": phase,
                    "runtime_seconds": experiment_result.get("runtime_seconds", 0)
                },
                timestamp=pd.Timestamp.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            print(f"Config {config['config_id']} failed: {e}")
            return None
    
    def _save_results(self, results: list[ConfigResult], filename: str):
        """Save results with full provenance."""
        results_data = {
            "experiment_profile": asdict(self.profile),
            "timestamp": pd.Timestamp.now().isoformat(),
            "num_results": len(results),
            "results": [asdict(result) for result in results]
        }
        
        output_path = self.exp_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"💾 Saved {len(results)} results to {output_path}")
    
    def run_full_experiment(self) -> dict[str, Any]:
        """Run complete discovery -> confirmation experiment."""
        start_time = time.time()
        
        try:
            # Discovery phase
            discovery_results = self.run_discovery_phase()
            
            # Candidate selection
            candidates = self.select_candidates(discovery_results)
            
            # Confirmation phase
            confirmation_results = self.run_confirmation_phase(candidates)
            
            # Final report
            runtime = time.time() - start_time
            report = self._generate_final_report(discovery_results, confirmation_results, runtime)
            
            return report
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            raise
    
    def _generate_final_report(self, discovery: list[ConfigResult], confirmation: list[ConfigResult], runtime: float) -> dict[str, Any]:
        """Generate final experiment report."""
        report = {
            "experiment_name": self.profile.name,
            "hypothesis": self.profile.hypothesis,
            "runtime_minutes": runtime / 60,
            "discovery_phase": {
                "configs_tested": len(discovery),
                "mean_ic": np.mean([r.ic_stats.mean_ic for r in discovery]) if discovery else 0,
                "mean_sharpe": np.mean([r.sharpe for r in discovery]) if discovery else 0,
                "best_ic": max([r.ic_stats.mean_ic for r in discovery]) if discovery else 0,
                "best_sharpe": max([r.sharpe for r in discovery]) if discovery else 0
            },
            "confirmation_phase": {
                "candidates_tested": len(confirmation),
                "mean_ic": np.mean([r.ic_stats.mean_ic for r in confirmation]) if confirmation else 0,
                "mean_sharpe": np.mean([r.sharpe for r in confirmation]) if confirmation else 0,
                "significant_results": len([r for r in confirmation if r.ic_stats.t_stat > 2.0]),
                "promoted_configs": [r.config_id for r in confirmation if r.ic_stats.t_stat > 2.0 and r.sharpe > self.profile.min_sharpe_threshold]
            },
            "timestamp": pd.Timestamp.now().isoformat(),
            "experiment_directory": str(self.exp_dir)
        }
        
        # Save final report
        with open(self.exp_dir / "final_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n🎉 EXPERIMENT COMPLETE")
        print(f"⏱️  Runtime: {runtime/60:.1f} minutes")
        print(f"🔍 Discovery: {len(discovery)} configs tested")
        print(f"🔒 Confirmation: {len(confirmation)} candidates validated")
        print(f"🏆 Promoted: {len(report['confirmation_phase']['promoted_configs'])} configs")
        print(f"📁 Full results: {self.exp_dir}")
        
        if report['confirmation_phase']['promoted_configs']:
            print(f"🚀 Ready for paper trading: {report['confirmation_phase']['promoted_configs']}")
        else:
            print("💡 No configs promoted. Consider: different features, longer discovery, or hypothesis revision.")
        
        return report


def main():
    """CLI for configuration sweep runner."""
    parser = argparse.ArgumentParser(description="Professional configuration sweep with statistical validation")
    parser.add_argument("profile", help="Path to experiment profile YAML")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't run")
    parser.add_argument("--discovery-only", action="store_true", help="Run discovery phase only")
    
    args = parser.parse_args()
    
    if not Path(args.profile).exists():
        print(f"❌ Profile not found: {args.profile}")
        return 1
    
    try:
        runner = ConfigSweepRunner(args.profile)
        
        if args.dry_run:
            configs = runner.generate_configs()
            print(f"Generated {len(configs)} configurations")
            return 0
        
        if args.discovery_only:
            results = runner.run_discovery_phase()
            candidates = runner.select_candidates(results)
            print(f"Discovery complete. {len(candidates)} candidates ready for confirmation.")
            return 0
        
        # Full experiment
        report = runner.run_full_experiment()
        return 0 if report['confirmation_phase']['promoted_configs'] else 1
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
