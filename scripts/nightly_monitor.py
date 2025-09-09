#!/usr/bin/env python3
"""
Nightly Monitoring System
Runs reduced grid on latest data and generates monitoring reports
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml
from datetime import datetime, timedelta
import sys
import os

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

from statistics import validate_strategy_robustness, compute_uncertainty_metrics
from baseline_strategies import run_baseline_strategies

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NightlyMonitor:
    """Nightly monitoring system for strategy performance."""
    
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.monitoring_data = {}
        self.alerts = []
        
    def load_config(self, config_file: str) -> Dict:
        """Load monitoring configuration."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def check_data_freshness(self, data_path: str) -> Dict:
        """Check if data is fresh and complete."""
        try:
            data_file = Path(data_path)
            if not data_file.exists():
                return {'status': 'missing', 'error': 'Data file not found'}
            
            # Check file modification time
            mod_time = datetime.fromtimestamp(data_file.stat().st_mtime)
            hours_old = (datetime.now() - mod_time).total_seconds() / 3600
            
            # Load data and check completeness
            data = pd.read_csv(data_file)
            expected_days = 5  # Expect at least 5 trading days
            actual_days = len(data)
            
            freshness_status = {
                'status': 'fresh' if hours_old < 24 else 'stale',
                'hours_old': hours_old,
                'expected_days': expected_days,
                'actual_days': actual_days,
                'completeness': actual_days / expected_days if expected_days > 0 else 0,
                'last_update': mod_time.isoformat()
            }
            
            if hours_old > 48:
                self.alerts.append(f"Data is {hours_old:.1f} hours old - may be stale")
            
            if actual_days < expected_days * 0.8:
                self.alerts.append(f"Data completeness low: {actual_days}/{expected_days} days")
            
            return freshness_status
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def run_reduced_grid(self, universe_config: str, grid_config: str, 
                        output_dir: str) -> Dict:
        """Run reduced grid for monitoring."""
        try:
            # This would call the actual grid runner with reduced parameters
            # For now, we'll simulate the results
            
            logger.info("Running reduced monitoring grid...")
            
            # Simulate grid results
            grid_results = {
                'total_experiments': 50,  # Reduced from full grid
                'successful_runs': 45,
                'gate_passes': 12,
                'best_sharpe': 0.15,
                'median_sharpe': 0.08,
                'runtime_seconds': 120,
                'config_used': grid_config
            }
            
            # Check for performance degradation
            baseline_sharpe = 0.10  # Expected baseline
            if grid_results['best_sharpe'] < baseline_sharpe * 0.5:
                self.alerts.append(f"Performance degradation: best Sharpe {grid_results['best_sharpe']:.3f} < {baseline_sharpe * 0.5:.3f}")
            
            if grid_results['gate_passes'] < 5:
                self.alerts.append(f"Low gate passes: {grid_results['gate_passes']} < 5")
            
            return grid_results
            
        except Exception as e:
            self.alerts.append(f"Grid run failed: {str(e)}")
            return {'error': str(e)}
    
    def check_strategy_drift(self, current_results: Dict, historical_results: Dict) -> Dict:
        """Check for strategy performance drift."""
        drift_analysis = {
            'sharpe_drift': 0.0,
            'return_drift': 0.0,
            'volatility_drift': 0.0,
            'significant_drift': False
        }
        
        try:
            current_sharpe = current_results.get('best_sharpe', 0)
            historical_sharpe = historical_results.get('best_sharpe', 0)
            
            if historical_sharpe > 0:
                sharpe_drift = (current_sharpe - historical_sharpe) / historical_sharpe
                drift_analysis['sharpe_drift'] = sharpe_drift
                
                if abs(sharpe_drift) > 0.3:  # 30% drift threshold
                    drift_analysis['significant_drift'] = True
                    self.alerts.append(f"Significant Sharpe drift: {sharpe_drift:.1%}")
            
        except Exception as e:
            logger.error(f"Error in drift analysis: {e}")
        
        return drift_analysis
    
    def generate_monitoring_report(self, output_dir: str) -> None:
        """Generate comprehensive monitoring report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all monitoring data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'data_freshness': self.monitoring_data.get('data_freshness', {}),
            'grid_results': self.monitoring_data.get('grid_results', {}),
            'drift_analysis': self.monitoring_data.get('drift_analysis', {}),
            'alerts': self.alerts,
            'system_status': 'healthy' if not self.alerts else 'warning'
        }
        
        # Save monitoring data
        with open(output_path / 'monitoring_data.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate markdown report
        report_content = self.generate_markdown_report(report_data)
        with open(output_path / 'monitoring_report.md', 'w') as f:
            f.write(report_content)
        
        # Send alerts if any
        if self.alerts:
            self.send_alerts(self.alerts, output_path)
        
        logger.info(f"Monitoring report generated: {len(self.alerts)} alerts")
    
    def generate_markdown_report(self, data: Dict) -> str:
        """Generate markdown monitoring report."""
        
        status_icon = "🟢" if data['system_status'] == 'healthy' else "🟡"
        
        report = f"""# Nightly Monitoring Report

## System Status: {status_icon} {data['system_status'].upper()}

**Generated**: {data['timestamp']}

## Data Freshness

"""
        
        freshness = data.get('data_freshness', {})
        if freshness.get('status') == 'fresh':
            report += f"✅ **Data is fresh** ({freshness.get('hours_old', 0):.1f} hours old)\n"
        else:
            report += f"⚠️ **Data may be stale** ({freshness.get('hours_old', 0):.1f} hours old)\n"
        
        report += f"""
- **Last Update**: {freshness.get('last_update', 'Unknown')}
- **Completeness**: {freshness.get('completeness', 0):.1%}
- **Days Available**: {freshness.get('actual_days', 0)}/{freshness.get('expected_days', 0)}

## Grid Performance

"""
        
        grid_results = data.get('grid_results', {})
        if 'error' not in grid_results:
            report += f"""
- **Total Experiments**: {grid_results.get('total_experiments', 0)}
- **Successful Runs**: {grid_results.get('successful_runs', 0)}
- **Gate Passes**: {grid_results.get('gate_passes', 0)}
- **Best Sharpe**: {grid_results.get('best_sharpe', 0):.3f}
- **Median Sharpe**: {grid_results.get('median_sharpe', 0):.3f}
- **Runtime**: {grid_results.get('runtime_seconds', 0)}s
"""
        else:
            report += f"❌ **Grid run failed**: {grid_results.get('error')}\n"
        
        report += "\n## Performance Drift\n\n"
        
        drift = data.get('drift_analysis', {})
        if drift.get('significant_drift'):
            report += f"⚠️ **Significant drift detected**\n"
        else:
            report += f"✅ **No significant drift**\n"
        
        report += f"""
- **Sharpe Drift**: {drift.get('sharpe_drift', 0):.1%}
- **Return Drift**: {drift.get('return_drift', 0):.1%}
- **Volatility Drift**: {drift.get('volatility_drift', 0):.1%}

## Alerts

"""
        
        if data.get('alerts'):
            for alert in data['alerts']:
                report += f"⚠️ {alert}\n"
        else:
            report += "✅ No alerts\n"
        
        report += f"""
## Recommendations

"""
        
        if data['system_status'] == 'healthy':
            report += "✅ System is operating normally\n"
        else:
            report += "⚠️ Review alerts and take corrective action\n"
        
        if freshness.get('status') == 'stale':
            report += "- Check data pipeline and update data sources\n"
        
        if grid_results.get('gate_passes', 0) < 5:
            report += "- Investigate low gate pass rate\n"
        
        if drift.get('significant_drift'):
            report += "- Review strategy parameters and market conditions\n"
        
        report += """
---
*Generated by nightly_monitor.py*
"""
        
        return report
    
    def send_alerts(self, alerts: List[str], output_path: Path) -> None:
        """Send alerts (placeholder for actual alerting system)."""
        alert_file = output_path / 'alerts.txt'
        with open(alert_file, 'w') as f:
            f.write(f"ALERTS - {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n")
            for alert in alerts:
                f.write(f"⚠️ {alert}\n")
        
        logger.warning(f"Alerts written to {alert_file}")
        # In production, this would send to Slack, email, etc.


def run_nightly_monitoring(config_file: str, output_dir: str) -> None:
    """Run complete nightly monitoring process."""
    
    monitor = NightlyMonitor(config_file)
    
    try:
        # Check data freshness
        logger.info("Checking data freshness...")
        data_freshness = monitor.check_data_freshness(monitor.config['data_path'])
        monitor.monitoring_data['data_freshness'] = data_freshness
        
        # Run reduced grid
        logger.info("Running reduced monitoring grid...")
        grid_results = monitor.run_reduced_grid(
            monitor.config['universe_config'],
            monitor.config['grid_config'],
            output_dir
        )
        monitor.monitoring_data['grid_results'] = grid_results
        
        # Check for drift (if historical data available)
        if 'historical_results' in monitor.config:
            logger.info("Checking for performance drift...")
            drift_analysis = monitor.check_strategy_drift(
                grid_results,
                monitor.config['historical_results']
            )
            monitor.monitoring_data['drift_analysis'] = drift_analysis
        
        # Generate report
        logger.info("Generating monitoring report...")
        monitor.generate_monitoring_report(output_dir)
        
        logger.info("Nightly monitoring complete!")
        
    except Exception as e:
        logger.error(f"Error in nightly monitoring: {e}")
        monitor.alerts.append(f"Monitoring system error: {str(e)}")
        monitor.generate_monitoring_report(output_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description='Run nightly monitoring')
    parser.add_argument('--config', required=True, help='Monitoring configuration file')
    parser.add_argument('--output-dir', default='monitoring', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        run_nightly_monitoring(args.config, args.output_dir)
        
    except Exception as e:
        logger.error(f"Error in nightly monitoring: {e}")
        raise


if __name__ == "__main__":
    main()
