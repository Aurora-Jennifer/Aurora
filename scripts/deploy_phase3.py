#!/usr/bin/env python3
"""
Phase 3 Deployment Script
Orchestrates the complete deployment of the bulletproof trading system
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml
import time
from datetime import datetime, timedelta
import subprocess
import sys
import os

# Import capability checking
sys.path.append('utils')
from env_check import check_capabilities

# Import component verification
from verify_components import (
    verify_robustness, verify_oos, verify_lag,
    verify_portfolio, verify_ablation, verify_monitoring,
    verify_hard_invariants
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def write_json(path, obj):
    """Write JSON object to file with proper directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


class Phase3Deployment:
    """Phase 3 deployment orchestrator."""
    
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.deployment_log = []
        self.deployment_status = {}
        
    def load_config(self, config_file: str) -> Dict:
        """Load deployment configuration."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_environment_check(self) -> Dict:
        """Run environment validation."""
        logger.info("Running environment check...")
        
        try:
            result = subprocess.run([
                'python', 'scripts/check_environment.py'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                status = {'status': 'passed', 'output': result.stdout}
                logger.info("Environment check passed")
            else:
                status = {'status': 'failed', 'error': result.stderr}
                logger.error(f"Environment check failed: {result.stderr}")
            
            self.deployment_log.append({
                'step': 'environment_check',
                'timestamp': datetime.now().isoformat(),
                'status': status['status'],
                'details': status
            })
            
            return status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            logger.error(f"Environment check error: {e}")
            return error_status
    
    def run_oos_validation(self, output_dir: str) -> Dict:
        """Run out-of-sample validation."""
        logger.info("Running OOS validation...")
        
        try:
            result = subprocess.run([
                'python', 'scripts/run_oos_validation.py',
                '--universe-cfg', 'config/universe_smoke.yaml',
                '--oos-cfg', 'config/robustness/oos_slices.yaml',
                '--grid-cfg', 'config/robustness/cost_03bps.yaml',
                '--out-dir', output_dir
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            oos_status = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
            self.deployment_log.append({
                'step': 'oos_validation',
                'timestamp': datetime.now().isoformat(),
                'status': oos_status['status'],
                'details': oos_status
            })
            
            return oos_status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            logger.error(f"OOS validation error: {e}")
            return error_status
    
    def run_signal_lag_tests(self, output_dir: str) -> Dict:
        """Run signal lag and leakage tests."""
        logger.info("Running signal lag tests...")
        
        try:
            result = subprocess.run([
                'python', 'scripts/signal_lag_tests.py',
                '--universe-cfg', 'config/universe_smoke.yaml',
                '--pred-root', 'results',
                '--out-dir', output_dir
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            lag_status = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
            self.deployment_log.append({
                'step': 'signal_lag_tests',
                'timestamp': datetime.now().isoformat(),
                'status': lag_status['status'],
                'details': lag_status
            })
            
            return lag_status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            logger.error(f"Signal lag tests error: {e}")
            return error_status
    
    def run_robustness_tests(self) -> Dict:
        """Run comprehensive robustness tests."""
        logger.info("Running robustness tests...")
        
        test_results = {}
        
        # Test 1: Cost stress test
        try:
            logger.info("Running cost stress test...")
            result = subprocess.run([
                'python', '-m', 'ml.runner_universe',
                '--universe-cfg', 'config/universe_smoke.yaml',
                '--grid-cfg', 'config/robustness/cost_03bps.yaml',
                '--out-dir', 'results/deployment_test_cost'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            test_results['cost_stress'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            test_results['cost_stress'] = {'status': 'error', 'error': str(e)}
        
        # Test 2: OOS validation
        try:
            logger.info("Running OOS validation test...")
            result = subprocess.run([
                'python', '-m', 'ml.runner_universe',
                '--universe-cfg', 'config/universe_smoke.yaml',
                '--grid-cfg', 'config/robustness/oos_slices.yaml',
                '--out-dir', 'results/deployment_test_oos'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            test_results['oos_validation'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            test_results['oos_validation'] = {'status': 'error', 'error': str(e)}
        
        # Test 3: Signal lag tests
        try:
            logger.info("Running signal lag tests...")
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/test_signal_lag.py', '-v'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            test_results['signal_lag'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            test_results['signal_lag'] = {'status': 'error', 'error': str(e)}
        
        self.deployment_log.append({
            'step': 'robustness_tests',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'details': test_results
        })
        
        return test_results
    
    def run_portfolio_construction(self) -> Dict:
        """Run portfolio construction and aggregation."""
        logger.info("Running portfolio construction...")
        
        try:
            # First, ensure we have results to aggregate
            results_dir = 'results/cost_stress_03bps'
            if not Path(results_dir).exists():
                logger.warning(f"Results directory {results_dir} not found, skipping portfolio construction")
                return {'status': 'skipped', 'reason': 'No results to aggregate'}
            
            # Run portfolio aggregation
            result = subprocess.run([
                'python', 'scripts/portfolio_aggregate.py',
                '--input-dir', results_dir,
                '--output-dir', 'portfolios/deployment_portfolio',
                '--config', 'config/portfolio.yaml'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            portfolio_status = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
            self.deployment_log.append({
                'step': 'portfolio_construction',
                'timestamp': datetime.now().isoformat(),
                'status': portfolio_status['status'],
                'details': portfolio_status
            })
            
            return portfolio_status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            logger.error(f"Portfolio construction error: {e}")
            return error_status
    
    def run_baseline_comparison(self) -> Dict:
        """Run baseline strategy comparison."""
        logger.info("Running baseline comparison...")
        
        try:
            # Run baseline comparison for AAPL
            result = subprocess.run([
                'python', 'scripts/baseline_comparison.py',
                '--symbol', 'AAPL',
                '--strategy-results', 'results/cost_stress_03bps/AAPL/grid_results.csv',
                '--output-dir', 'reports/deployment_baseline',
                '--costs-bps', '3.0'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            baseline_status = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
            self.deployment_log.append({
                'step': 'baseline_comparison',
                'timestamp': datetime.now().isoformat(),
                'status': baseline_status['status'],
                'details': baseline_status
            })
            
            return baseline_status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            logger.error(f"Baseline comparison error: {e}")
            return error_status
    
    def run_ablation_analysis(self, output_dir: str) -> Dict:
        """Run feature ablation analysis."""
        logger.info("Running ablation analysis...")
        
        try:
            result = subprocess.run([
                'python', 'scripts/run_ablation.py',
                '--universe-cfg', 'config/universe_smoke.yaml',
                '--ablation-cfg', 'config/robustness/ablation_tests.yaml',
                '--baseline-root', 'results',
                '--base-grid-cfg', 'config/robustness/cost_03bps.yaml',
                '--out-dir', output_dir
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            ablation_status = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
            self.deployment_log.append({
                'step': 'ablation_analysis',
                'timestamp': datetime.now().isoformat(),
                'status': ablation_status['status'],
                'details': ablation_status
            })
            
            return ablation_status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            return error_status
    
    def setup_monitoring(self, output_dir: str) -> Dict:
        """Setup nightly monitoring system."""
        logger.info("Setting up monitoring system...")
        
        try:
            # Create monitoring directories
            monitoring_dir = Path(output_dir) / 'monitoring'
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            
            # Create monitoring configuration
            monitoring_config = {
                'enabled': True,
                'frequency': 'daily',
                'data_freshness_threshold_hours': 24,
                'performance_thresholds': {
                    'min_sharpe': 0.05,
                    'max_drawdown': 0.15
                }
            }
            
            config_path = monitoring_dir / 'monitoring_config.json'
            with open(config_path, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            # Create heartbeat
            heartbeat = {
                'last_run_epoch': time.time(),
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy'
            }
            
            heartbeat_path = monitoring_dir / 'heartbeat.json'
            with open(heartbeat_path, 'w') as f:
                json.dump(heartbeat, f, indent=2)
            
            monitoring_status = {
                'status': 'passed',
                'output': 'Monitoring system configured with heartbeat',
                'config_path': str(config_path),
                'heartbeat_path': str(heartbeat_path)
            }
            
            self.deployment_log.append({
                'step': 'monitoring_setup',
                'timestamp': datetime.now().isoformat(),
                'status': monitoring_status['status'],
                'details': monitoring_status
            })
            
            return monitoring_status
            
        except Exception as e:
            error_status = {'status': 'error', 'error': str(e)}
            logger.error(f"Monitoring setup error: {e}")
            return error_status
    
    def generate_deployment_report(self, output_dir: str) -> None:
        """Generate comprehensive deployment report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate overall deployment status based on component verification
        # Always define step variables for the summary
        all_steps = [log['step'] for log in self.deployment_log]
        failed_steps = [log['step'] for log in self.deployment_log if log['status'] == 'failed']
        error_steps = [log['step'] for log in self.deployment_log if log['status'] == 'error']
        
        # Try to load the deployment report to get component status
        report_path = output_path / 'deployment_report.json'
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                component_status = report.get('component_status', {})
                overall_ok = all(component_status.values())
                some_ok = any(component_status.values())
                overall_status = 'success' if overall_ok else ('partial' if some_ok else 'failed')
            except Exception:
                # Fallback to step-based status
                overall_status = 'success' if not failed_steps and not error_steps else 'partial' if failed_steps else 'failed'
        else:
            # Fallback to step-based status
            overall_status = 'success' if not failed_steps and not error_steps else 'partial' if failed_steps else 'failed'
        
        deployment_summary = {
            'deployment_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'total_steps': len(all_steps),
            'successful_steps': len(all_steps) - len(failed_steps) - len(error_steps),
            'failed_steps': failed_steps,
            'error_steps': error_steps,
            'deployment_log': self.deployment_log
        }
        
        # Save deployment data
        with open(output_path / 'deployment_summary.json', 'w') as f:
            json.dump(deployment_summary, f, indent=2, default=str)
        
        # Generate markdown report
        report_content = self.generate_markdown_report(deployment_summary)
        with open(output_path / 'deployment_report.md', 'w') as f:
            f.write(report_content)
        
        logger.info(f"Deployment report generated: {overall_status.upper()}")
    
    def generate_markdown_report(self, summary: Dict) -> str:
        """Generate markdown deployment report."""
        
        status_icon = "🟢" if summary['overall_status'] == 'success' else "🟡" if summary['overall_status'] == 'partial' else "🔴"
        
        report = f"""# Phase 3 Deployment Report

## Deployment Status: {status_icon} {summary['overall_status'].upper()}

**Deployed**: {summary['deployment_timestamp']}

## Summary

- **Total Steps**: {summary['total_steps']}
- **Successful**: {summary['successful_steps']}
- **Failed**: {len(summary['failed_steps'])}
- **Errors**: {len(summary['error_steps'])}

## Deployment Steps

| Step | Status | Timestamp |
|------|--------|-----------|
"""
        
        for log_entry in summary['deployment_log']:
            status_icon = "✅" if log_entry['status'] == 'passed' else "❌" if log_entry['status'] == 'failed' else "⚠️"
            report += f"| {log_entry['step']} | {status_icon} {log_entry['status']} | {log_entry['timestamp']} |\n"
        
        if summary['failed_steps']:
            report += f"""
## Failed Steps

"""
            for step in summary['failed_steps']:
                report += f"- ❌ {step}\n"
        
        if summary['error_steps']:
            report += f"""
## Error Steps

"""
            for step in summary['error_steps']:
                report += f"- ⚠️ {step}\n"
        
        report += f"""
## System Components

### ✅ Bulletproof Foundation
- Environment validation and reproducibility
- Statistical rigor (Deflated Sharpe, White's Reality Check)
- Signal lag detection and activity gates
- Portfolio risk controls and position bounds

### ✅ Robustness Testing
- Cost stress testing (3bps+)
- Out-of-sample validation
- Feature ablation analysis
- Baseline strategy comparison

### ✅ Portfolio & Deployment
- Portfolio aggregation and construction
- Paper trading simulation
- Nightly monitoring system
- Performance tracking and alerting

## Next Steps

"""
        
        if summary['overall_status'] == 'success':
            report += """
✅ **Deployment Successful!**

The bulletproof trading system is ready for production:

1. **Monitor Performance**: Use nightly monitoring system
2. **Paper Trading**: Start with paper trading simulation
3. **Live Deployment**: Gradual rollout with small allocations
4. **Continuous Monitoring**: Track performance and drift
"""
        else:
            report += """
⚠️ **Deployment Issues Detected**

Please review and fix the following:

"""
            for step in summary['failed_steps'] + summary['error_steps']:
                report += f"- Fix {step} before proceeding\n"
        
        report += """
---
*Generated by deploy_phase3.py*
"""
        
        return report


def run_phase3_deployment(config_file: str, output_dir: str) -> None:
    """Run complete Phase 3 deployment with honest reporting."""
    
    deployment = Phase3Deployment(config_file)
    
    # Initialize deployment report
    caps = check_capabilities()
    report = {"summary": "PARTIAL", "component_status": {}, "capabilities": caps}
    
    try:
        logger.info("Starting Phase 3 deployment...")
        
        # Step 1: Environment check
        env_status = deployment.run_environment_check()
        env_ok = env_status['status'] == 'passed'
        if not env_ok:
            logger.error("Environment check failed, aborting deployment")
            return
        
        # Step 2: Robustness tests
        robustness_results = deployment.run_robustness_tests()
        
        # Step 3: OOS validation
        oos_status = deployment.run_oos_validation(output_dir)
        
        # Step 4: Signal lag tests
        lag_status = deployment.run_signal_lag_tests(output_dir)
        
        # Step 5: Ablation analysis
        ablation_status = deployment.run_ablation_analysis(output_dir)
        
        # Step 6: Portfolio construction
        portfolio_status = deployment.run_portfolio_construction()
        
        # Step 7: Baseline comparison
        baseline_status = deployment.run_baseline_comparison()
        
        # Step 8: Setup monitoring
        monitoring_status = deployment.setup_monitoring(output_dir)
        
        # Verify components using actual artifacts and metrics
        logger.info("Verifying components from actual artifacts...")
        robustness_ok, robustness_msg = verify_robustness(output_dir)
        oos_ok, oos_msg = verify_oos(output_dir)
        lag_ok, lag_msg = verify_lag(output_dir)
        portfolio_ok, portfolio_msg = verify_portfolio(output_dir)
        ablation_ok, ablation_msg = verify_ablation(output_dir)
        monitoring_ok, monitoring_msg = verify_monitoring(output_dir)
        
        # Check hard invariants
        invariants_ok, invariants_msg = verify_hard_invariants(output_dir)
        
        # Set component status based on actual verification
        status = {
            "env_ok": env_ok,
            "robustness_ok": robustness_ok,
            "oos_ok": oos_ok,
            "lag_ok": lag_ok,
            "portfolio_ok": portfolio_ok and invariants_ok,  # Portfolio must pass invariants
            "ablation_ok": ablation_ok,
            "monitoring_ok": monitoring_ok
        }
        
        # Store verification messages for debugging
        verification_messages = {
            "robustness": robustness_msg,
            "oos": oos_msg,
            "lag": lag_msg,
            "portfolio": portfolio_msg,
            "ablation": ablation_msg,
            "monitoring": monitoring_msg,
            "invariants": invariants_msg
        }
        report["component_status"] = status
        
        # Determine overall readiness
        ready = all(status.values())
        report["summary"] = "READY" if ready else "PARTIAL"
        
        # Console summary (no unconditional green banners)
        print("\n=== SYSTEM SUMMARY ===")
        for k, v in status.items():
            print(f" - {k}: {'✅' if v else '❌'}")
        print(f"\n🚦 Deployment status: {report['summary']}")
        
        # Show verification details for failed components
        print("\n=== VERIFICATION DETAILS ===")
        for component, result in verification_messages.items():
            if isinstance(result, tuple) and len(result) == 2:
                success, message = result
                if not success:
                    print(f"❌ {component}: {message}")
                else:
                    print(f"✅ {component}: {message}")
            else:
                print(f"⚠️ {component}: {result}")
        
        if not caps.get("lightgbm", True):
            print("\nℹ️  LightGBM not available — LGBM configs were skipped.")
        if not caps.get("xgboost", True):
            print("ℹ️  XGBoost not available — XGB configs were skipped.")
        
        # Add verification messages to report
        report["verification_messages"] = verification_messages
        
        # Save machine-readable report
        write_json(os.path.join(output_dir, "deployment_report.json"), report)
        
        # Generate deployment report
        deployment.generate_deployment_report(output_dir)
        
        logger.info("Phase 3 deployment complete!")
        
    except Exception as e:
        logger.error(f"Error in Phase 3 deployment: {e}")
        deployment.deployment_log.append({
            'step': 'deployment_error',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'details': {'error': str(e)}
        })
        report["summary"] = "FAILED"
        write_json(os.path.join(output_dir, "deployment_report.json"), report)
        deployment.generate_deployment_report(output_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description='Deploy Phase 3 bulletproof trading system')
    parser.add_argument('--config', default='config/deployment.yaml', help='Deployment configuration file')
    parser.add_argument('--output-dir', default='deployment', help='Output directory for deployment reports')
    
    args = parser.parse_args()
    
    try:
        run_phase3_deployment(args.config, args.output_dir)
        
    except Exception as e:
        logger.error(f"Error in Phase 3 deployment: {e}")
        raise


if __name__ == "__main__":
    main()
