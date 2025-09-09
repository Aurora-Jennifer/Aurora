"""
Paper trading automated reporting system.

Generates daily/weekly reports with performance metrics and alerts.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings


class PaperTradingReporter:
    """
    Automated reporting for paper trading with alerting.
    """
    
    def __init__(self, 
                 reports_dir: str = "results/paper/reports",
                 alert_thresholds: Dict = None):
        """
        Initialize paper trading reporter.
        
        Args:
            reports_dir: Directory to store reports
            alert_thresholds: Alert threshold configuration
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'entropy_floor': 0.75,
            'entropy_consecutive_bars': 10,
            'pnl_divergence_sigma': 3.0,
            'blocked_order_pct': 5.0,
            'max_drawdown_pct': 15.0,
            'min_daily_ic': -0.1,
            'max_daily_ic': 0.1
        }
        
        self.alerts_triggered = []
    
    def generate_daily_report(self,
                             date: str,
                             performance_metrics: Dict,
                             trading_metrics: Dict,
                             risk_metrics: Dict,
                             system_metrics: Dict) -> Dict:
        """
        Generate daily paper trading report.
        
        Args:
            date: Trading date (YYYY-MM-DD)
            performance_metrics: Dict with IC, Sharpe, returns, etc.
            trading_metrics: Dict with turnover, positions, trades, etc.
            risk_metrics: Dict with exposures, drawdown, VaR, etc.
            system_metrics: Dict with execution, slippage, etc.
            
        Returns:
            Dict with complete daily report
        """
        print(f"📊 Generating daily report for {date}...")
        
        # Core daily metrics
        daily_report = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            
            # Performance metrics
            'daily_ic': performance_metrics.get('ic', 0.0),
            'daily_returns_pct': performance_metrics.get('returns_pct', 0.0),
            'cumulative_returns_pct': performance_metrics.get('cumulative_returns_pct', 0.0),
            'decile_spread': performance_metrics.get('decile_spread', 0.0),
            'hit_rate': performance_metrics.get('hit_rate', 0.0),
            'sharpe_ytd': performance_metrics.get('sharpe_ytd', 0.0),
            
            # Trading metrics
            'turnover': trading_metrics.get('turnover', 0.0),
            'num_positions': trading_metrics.get('num_positions', 0),
            'num_trades': trading_metrics.get('num_trades', 0),
            'gross_exposure_pct': trading_metrics.get('gross_exposure_pct', 0.0),
            'net_exposure_pct': trading_metrics.get('net_exposure_pct', 0.0),
            
            # Risk metrics
            'max_drawdown_pct': risk_metrics.get('max_drawdown_pct', 0.0),
            'daily_var_pct': risk_metrics.get('daily_var_pct', 0.0),
            'size_factor_exposure': risk_metrics.get('size_factor_exposure', 0.0),
            'sector_factor_exposure': risk_metrics.get('sector_factor_exposure', 0.0),
            'market_beta': risk_metrics.get('market_beta', 0.0),
            
            # System metrics
            'realized_slippage_bps': system_metrics.get('realized_slippage_bps', 0.0),
            'expected_slippage_bps': system_metrics.get('expected_slippage_bps', 0.0),
            'blocked_order_pct': system_metrics.get('blocked_order_pct', 0.0),
            'fill_rate_pct': system_metrics.get('fill_rate_pct', 100.0),
            'execution_time_ms': system_metrics.get('execution_time_ms', 0.0),
            'action_entropy': system_metrics.get('action_entropy', 1.0),
            
            # Alerts
            'alerts_triggered': [],
            'guard_breaches': system_metrics.get('guard_breaches', [])
        }
        
        # Check for alerts
        alerts = self._check_daily_alerts(daily_report)
        daily_report['alerts_triggered'] = alerts
        
        # Save daily report
        daily_file = self.reports_dir / f"daily_{date}.json"
        with open(daily_file, 'w') as f:
            json.dump(daily_report, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        self._save_daily_csv(daily_report)
        
        print(f"✅ Daily report saved: {daily_file}")
        if alerts:
            print(f"⚠️ Alerts triggered: {len(alerts)}")
            for alert in alerts:
                print(f"   - {alert['type']}: {alert['message']}")
        
        return daily_report
    
    def generate_weekly_report(self, end_date: str) -> Dict:
        """
        Generate weekly rollup report.
        
        Args:
            end_date: End date for week (YYYY-MM-DD)
            
        Returns:
            Dict with weekly report
        """
        print(f"📈 Generating weekly report ending {end_date}...")
        
        # Load last 7 daily reports
        end_dt = pd.to_datetime(end_date)
        daily_reports = []
        
        for i in range(7):
            date = (end_dt - timedelta(days=i)).strftime('%Y-%m-%d')
            daily_file = self.reports_dir / f"daily_{date}.json"
            
            if daily_file.exists():
                with open(daily_file, 'r') as f:
                    daily_reports.append(json.load(f))
        
        if not daily_reports:
            print(f"⚠️ No daily reports found for week ending {end_date}")
            return {}
        
        # Calculate weekly aggregates
        weekly_report = {
            'week_ending': end_date,
            'trading_days': len(daily_reports),
            'timestamp': datetime.now().isoformat(),
            
            # Performance aggregates
            'avg_daily_ic': np.mean([r.get('daily_ic', 0) for r in daily_reports]),
            'weekly_returns_pct': sum(r.get('daily_returns_pct', 0) for r in daily_reports),
            'avg_decile_spread': np.mean([r.get('decile_spread', 0) for r in daily_reports]),
            'weekly_hit_rate': np.mean([r.get('hit_rate', 0) for r in daily_reports]),
            'weekly_sharpe': daily_reports[-1].get('sharpe_ytd', 0) if daily_reports else 0,
            
            # Trading aggregates
            'avg_turnover': np.mean([r.get('turnover', 0) for r in daily_reports]),
            'total_trades': sum(r.get('num_trades', 0) for r in daily_reports),
            'avg_gross_exposure': np.mean([r.get('gross_exposure_pct', 0) for r in daily_reports]),
            
            # Risk aggregates
            'max_drawdown_week': max(r.get('max_drawdown_pct', 0) for r in daily_reports),
            'avg_size_exposure': np.mean([r.get('size_factor_exposure', 0) for r in daily_reports]),
            'avg_sector_exposure': np.mean([r.get('sector_factor_exposure', 0) for r in daily_reports]),
            
            # System aggregates
            'avg_realized_slippage': np.mean([r.get('realized_slippage_bps', 0) for r in daily_reports]),
            'avg_expected_slippage': np.mean([r.get('expected_slippage_bps', 0) for r in daily_reports]),
            'avg_blocked_orders': np.mean([r.get('blocked_order_pct', 0) for r in daily_reports]),
            'total_alerts': sum(len(r.get('alerts_triggered', [])) for r in daily_reports),
            'total_guard_breaches': sum(len(r.get('guard_breaches', [])) for r in daily_reports)
        }
        
        # Check weekly alerts
        weekly_alerts = self._check_weekly_alerts(weekly_report, daily_reports)
        weekly_report['weekly_alerts'] = weekly_alerts
        
        # Performance vs expectations
        expected_metrics = {
            'expected_ic': 0.017,
            'expected_sharpe': 0.32,
            'expected_turnover': 1.8
        }
        
        weekly_report['vs_expectations'] = {
            'ic_vs_expected': weekly_report['avg_daily_ic'] - expected_metrics['expected_ic'],
            'sharpe_vs_expected': weekly_report['weekly_sharpe'] - expected_metrics['expected_sharpe'],
            'turnover_vs_expected': weekly_report['avg_turnover'] * 20 - expected_metrics['expected_turnover']  # Annualized
        }
        
        # Save weekly report
        weekly_file = self.reports_dir / f"weekly_{end_date}.json"
        with open(weekly_file, 'w') as f:
            json.dump(weekly_report, f, indent=2, default=str)
        
        print(f"✅ Weekly report saved: {weekly_file}")
        if weekly_alerts:
            print(f"🚨 Weekly alerts: {len(weekly_alerts)}")
            for alert in weekly_alerts:
                print(f"   - {alert['type']}: {alert['message']}")
        
        return weekly_report
    
    def _check_daily_alerts(self, daily_report: Dict) -> List[Dict]:
        """Check daily metrics against alert thresholds."""
        alerts = []
        
        # Check action entropy floor
        entropy = daily_report.get('action_entropy', 1.0)
        if entropy < self.alert_thresholds['entropy_floor']:
            alerts.append({
                'type': 'entropy_floor',
                'message': f'Action entropy {entropy:.3f} below {self.alert_thresholds["entropy_floor"]}',
                'severity': 'warning'
            })
        
        # Check blocked orders
        blocked_pct = daily_report.get('blocked_order_pct', 0.0)
        if blocked_pct > self.alert_thresholds['blocked_order_pct']:
            alerts.append({
                'type': 'blocked_orders',
                'message': f'Blocked orders {blocked_pct:.1f}% above {self.alert_thresholds["blocked_order_pct"]}%',
                'severity': 'warning'
            })
        
        # Check daily IC range
        daily_ic = daily_report.get('daily_ic', 0.0)
        if daily_ic < self.alert_thresholds['min_daily_ic'] or daily_ic > self.alert_thresholds['max_daily_ic']:
            alerts.append({
                'type': 'daily_ic_range',
                'message': f'Daily IC {daily_ic:.4f} outside expected range',
                'severity': 'info'
            })
        
        # Check drawdown
        drawdown = daily_report.get('max_drawdown_pct', 0.0)
        if drawdown > self.alert_thresholds['max_drawdown_pct']:
            alerts.append({
                'type': 'max_drawdown',
                'message': f'Drawdown {drawdown:.1f}% above {self.alert_thresholds["max_drawdown_pct"]}%',
                'severity': 'critical'
            })
        
        return alerts
    
    def _check_weekly_alerts(self, weekly_report: Dict, daily_reports: List[Dict]) -> List[Dict]:
        """Check weekly aggregates for alerts."""
        alerts = []
        
        # Check PnL divergence (simplified - would use actual variance in production)
        expected_weekly_return = 0.017 * len(daily_reports) / 252 * 100  # Annualized to weekly %
        actual_weekly_return = weekly_report.get('weekly_returns_pct', 0.0)
        weekly_std = 0.5  # Simplified - would calculate from historical data
        
        divergence_sigma = abs(actual_weekly_return - expected_weekly_return) / weekly_std
        if divergence_sigma > self.alert_thresholds['pnl_divergence_sigma']:
            alerts.append({
                'type': 'pnl_divergence',
                'message': f'Weekly PnL divergence {divergence_sigma:.1f}σ above {self.alert_thresholds["pnl_divergence_sigma"]}σ',
                'severity': 'warning'
            })
        
        # Check entropy consecutive bars
        entropy_violations = sum(1 for r in daily_reports 
                               if r.get('action_entropy', 1.0) < self.alert_thresholds['entropy_floor'])
        if entropy_violations >= self.alert_thresholds['entropy_consecutive_bars']:
            alerts.append({
                'type': 'entropy_consecutive',
                'message': f'Entropy below floor for {entropy_violations} consecutive days',
                'severity': 'critical'
            })
        
        return alerts
    
    def _save_daily_csv(self, daily_report: Dict):
        """Save daily report to CSV for easy analysis."""
        csv_file = self.reports_dir / "daily_reports.csv"
        
        # Create DataFrame from daily report (flatten nested dicts)
        flat_report = {}
        for key, value in daily_report.items():
            if isinstance(value, (list, dict)):
                continue  # Skip complex nested structures for CSV
            flat_report[key] = value
        
        df = pd.DataFrame([flat_report])
        
        # Append to existing CSV or create new
        if csv_file.exists():
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.drop_duplicates(subset=['date'], keep='last').to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, index=False)


def create_mock_paper_trading_data(num_days: int = 20) -> List[Dict]:
    """Create mock paper trading data for testing."""
    np.random.seed(42)
    
    reports = []
    cumulative_return = 0.0
    max_dd = 0.0
    
    for i in range(num_days):
        date = (datetime.now() - timedelta(days=num_days-i-1)).strftime('%Y-%m-%d')
        
        # Simulate realistic daily metrics
        daily_return = np.random.normal(0.08/252, 0.15/np.sqrt(252))  # Daily return with drift
        cumulative_return = (1 + cumulative_return) * (1 + daily_return) - 1
        
        # Simulate drawdown
        if cumulative_return < max_dd:
            max_dd = cumulative_return
        
        # Performance metrics
        performance = {
            'ic': np.random.normal(0.017, 0.03),
            'returns_pct': daily_return * 100,
            'cumulative_returns_pct': cumulative_return * 100,
            'decile_spread': np.random.normal(0.024, 0.008),
            'hit_rate': np.random.uniform(0.45, 0.55),
            'sharpe_ytd': cumulative_return / (0.15 * np.sqrt(i/252)) if i > 0 else 0
        }
        
        # Trading metrics
        trading = {
            'turnover': np.random.uniform(0.05, 0.12),
            'num_positions': np.random.randint(45, 55),
            'num_trades': np.random.randint(8, 25),
            'gross_exposure_pct': np.random.uniform(25, 32),
            'net_exposure_pct': np.random.uniform(-2, 2)
        }
        
        # Risk metrics
        risk = {
            'max_drawdown_pct': abs(max_dd) * 100,
            'daily_var_pct': np.random.uniform(0.8, 1.5),
            'size_factor_exposure': np.random.normal(0, 0.05),
            'sector_factor_exposure': np.random.normal(0, 0.03),
            'market_beta': np.random.normal(0, 0.1)
        }
        
        # System metrics
        system = {
            'realized_slippage_bps': np.random.uniform(4, 8),
            'expected_slippage_bps': 6.5,
            'blocked_order_pct': np.random.uniform(0, 3),
            'fill_rate_pct': np.random.uniform(95, 100),
            'execution_time_ms': np.random.uniform(50, 200),
            'action_entropy': np.random.uniform(0.7, 0.95),
            'guard_breaches': []
        }
        
        # Occasionally add alerts/breaches
        if np.random.random() < 0.1:  # 10% chance
            system['guard_breaches'] = ['position_concentration']
        
        reports.append({
            'date': date,
            'performance': performance,
            'trading': trading,
            'risk': risk,
            'system': system
        })
    
    return reports


def test_paper_trading_reports():
    """Test paper trading reporting system."""
    print("🧪 TESTING PAPER TRADING REPORTS")
    print("="*50)
    
    # Initialize reporter
    reporter = PaperTradingReporter("test_reports")
    
    # Create mock data
    mock_data = create_mock_paper_trading_data(14)  # 2 weeks
    
    print(f"✅ Created {len(mock_data)} days of mock data")
    
    # Generate daily reports
    for data in mock_data:
        daily_report = reporter.generate_daily_report(
            data['date'],
            data['performance'],
            data['trading'],
            data['risk'],
            data['system']
        )
    
    # Generate weekly report
    end_date = mock_data[-1]['date']
    weekly_report = reporter.generate_weekly_report(end_date)
    
    print(f"\n📊 WEEKLY SUMMARY:")
    print(f"   Avg Daily IC: {weekly_report.get('avg_daily_ic', 0):.4f}")
    print(f"   Weekly Returns: {weekly_report.get('weekly_returns_pct', 0):.2f}%")
    print(f"   Avg Turnover: {weekly_report.get('avg_turnover', 0):.3f}")
    print(f"   Total Alerts: {weekly_report.get('total_alerts', 0)}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_reports"):
        shutil.rmtree("test_reports")
    
    print(f"\n✅ Paper trading reports test completed")


if __name__ == "__main__":
    test_paper_trading_reports()
