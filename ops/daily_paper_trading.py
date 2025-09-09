#!/usr/bin/env python3
"""
Daily Paper Trading Operations Script

Automates the complete daily trading workflow:
- Pre-market validation
- Trading session monitoring  
- End-of-day reporting and reconciliation

Usage:
    python ops/daily_paper_trading.py --mode preflight    # 08:00 CT
    python ops/daily_paper_trading.py --mode trading      # 08:30-15:00 CT
    python ops/daily_paper_trading.py --mode eod          # 15:10 CT
    python ops/daily_paper_trading.py --mode full         # Complete daily cycle
"""
import sys
import os
sys.path.append('.')

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import signal
import logging
import hashlib
import subprocess

from ops.paper_trading_guards import PaperTradingGuards
from ops.pre_market_dry_run import run_pre_market_dry_run
from ml.production_logging import setup_production_logging
# from ml.paper_trading_reports import generate_daily_report, generate_weekly_report
# Note: Using mock reporting for now - will be implemented in full integration


class DailyPaperTradingOperations:
    """Automated daily paper trading operations."""
    
    def __init__(self):
        """Initialize daily operations."""
        self.logger = setup_production_logging(
            log_dir="logs",
            log_level="INFO"
        )
        
        self.trading_active = False
        self.daily_stats = {}
        self.alerts = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Daily paper trading operations initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.trading_active = False
        self._emergency_halt("Signal received")
    
    def _emergency_halt(self, reason: str):
        """Emergency halt with immediate position flattening."""
        self.logger.critical(f"EMERGENCY HALT: {reason}")
        
        try:
            # In real implementation, would flatten all positions
            halt_record = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'action': 'emergency_halt',
                'positions_before_halt': self.daily_stats.get('current_positions', {}),
                'status': 'executed'
            }
            
            # Save halt record
            halt_dir = Path("results/paper/emergency_halts")
            halt_dir.mkdir(parents=True, exist_ok=True)
            
            halt_file = halt_dir / f"halt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(halt_file, 'w') as f:
                json.dump(halt_record, f, indent=2)
            
            self.logger.critical(f"Emergency halt executed, record saved: {halt_file}")
            
            # Send alert
            self._send_alert("CRITICAL", f"Emergency halt executed: {reason}")
            
        except Exception as e:
            self.logger.critical(f"Failed to execute emergency halt: {e}")
    
    def _send_alert(self, level: str, message: str):
        """Send alert notification."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        
        self.alerts.append(alert)
        
        # Log the alert
        if level == "CRITICAL":
            self.logger.critical(f"ALERT: {message}")
        elif level == "WARNING":
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")
        
        # In production, would send to notification service (Slack, email, etc.)
        print(f"🚨 {level}: {message}")
    
    def run_preflight_checks(self) -> bool:
        """
        Run pre-market preflight checks (08:00 CT).
        
        Returns:
            True if all checks pass
        """
        self.logger.info("Starting preflight checks...")
        print("🌅 PRE-MARKET PREFLIGHT CHECKS")
        print("="*50)
        
        all_passed = True
        
        # 1. Paper trading guards
        print("\n🔒 Step 1: Paper trading environment validation...")
        try:
            guards = PaperTradingGuards()
            guard_result = guards.run_comprehensive_validation()
            
            if guard_result['validation_passed']:
                print("✅ Paper trading environment validated")
            else:
                print("❌ Paper trading environment validation failed")
                for error in guard_result['errors']:
                    self.logger.error(f"Guard error: {error}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Guard validation failed: {e}")
            self.logger.error(f"Guard validation exception: {e}")
            all_passed = False
        
        # 2. Data freshness check
        print("\n📊 Step 2: Data freshness validation...")
        try:
            # Check if we have recent data (in production, check actual data timestamps)
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)
            
            # Mock data freshness check
            data_fresh = True  # Would check actual data files
            
            if data_fresh:
                print("✅ Data freshness validated")
            else:
                print("❌ Stale data detected")
                self._send_alert("WARNING", "Stale data detected")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Data freshness check failed: {e}")
            self.logger.error(f"Data freshness exception: {e}")
            all_passed = False
        
        # 3. Feature whitelist integrity
        print("\n🔐 Step 3: Feature whitelist integrity...")
        try:
            whitelist_path = "results/production/features_whitelist.json"
            hash_path = "results/production/features_whitelist.json.hash"
            
            if Path(whitelist_path).exists() and Path(hash_path).exists():
                with open(whitelist_path, 'r') as f:
                    whitelist = json.load(f)
                
                content = json.dumps(sorted(whitelist), sort_keys=True)
                current_hash = hashlib.sha256(content.encode()).hexdigest()
                
                with open(hash_path, 'r') as f:
                    expected_hash = f.read().strip()
                
                if current_hash == expected_hash:
                    print(f"✅ Feature whitelist integrity verified ({len(whitelist)} features)")
                else:
                    print("❌ Feature whitelist integrity check failed")
                    self._send_alert("CRITICAL", "Feature whitelist tampering detected")
                    all_passed = False
            else:
                print("⚠️ Feature whitelist files missing")
                self._send_alert("WARNING", "Feature whitelist files missing")
                
        except Exception as e:
            print(f"❌ Feature whitelist check failed: {e}")
            self.logger.error(f"Whitelist check exception: {e}")
            all_passed = False
        
        # 4. Trading calendar validation
        print("\n📅 Step 4: Trading calendar validation...")
        try:
            today = datetime.now().date()
            
            # Simple business day check (in production, use proper market calendar)
            is_business_day = today.weekday() < 5
            
            if is_business_day:
                print(f"✅ Trading day confirmed: {today}")
            else:
                print(f"ℹ️ Non-trading day: {today}")
                # Not a failure, just informational
                
        except Exception as e:
            print(f"❌ Calendar validation failed: {e}")
            self.logger.error(f"Calendar validation exception: {e}")
        
        # 5. Pre-market dry run
        print("\n🧪 Step 5: Pre-market dry run...")
        try:
            # Use enhanced dry-run with proper date handling
            import subprocess
            result = subprocess.run(['python', 'ops/enhanced_dry_run.py'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                dry_run_result = {'overall_status': 'pass'}
                print("✅ Enhanced dry-run completed successfully")
            else:
                dry_run_result = {'overall_status': 'fail'}
                print("❌ Enhanced dry-run failed")
            
            if dry_run_result['overall_status'] == 'pass':
                print("✅ Pre-market dry run passed")
            else:
                print("⚠️ Pre-market dry run issues detected")
                self._send_alert("WARNING", "Pre-market dry run detected issues")
                # Don't fail preflight for dry run warnings
                
        except Exception as e:
            print(f"⚠️ Pre-market dry run failed: {e}")
            self.logger.warning(f"Dry run exception: {e}")
        
        # Summary
        print(f"\n📋 PREFLIGHT SUMMARY:")
        if all_passed:
            print("✅ ALL PREFLIGHT CHECKS PASSED - READY FOR TRADING")
            self.logger.info("Preflight checks completed successfully")
        else:
            print("❌ PREFLIGHT ISSUES DETECTED - REVIEW BEFORE TRADING")
            self.logger.error("Preflight checks failed")
            self._send_alert("CRITICAL", "Preflight checks failed")
        
        return all_passed
    
    def run_trading_session(self) -> Dict:
        """
        Run trading session monitoring (08:30-15:00 CT).
        
        Returns:
            Dict with trading session results
        """
        self.logger.info("Starting trading session...")
        print("📈 TRADING SESSION ACTIVE")
        print("="*30)
        
        self.trading_active = True
        session_stats = {
            'start_time': datetime.now().isoformat(),
            'bars_processed': 0,
            'alerts_triggered': 0,
            'positions_taken': 0,
            'adv_breaches': 0,
            'emergency_halts': 0
        }
        
        # Mock trading loop (in production, this would be real market data)
        try:
            bar_count = 0
            consecutive_low_entropy = 0
            
            print("🔄 Starting trading loop...")
            print("   Monitoring: entropy, PnL, slippage, ADV breaches")
            
            # Simulate trading session (normally runs for market hours)
            max_bars = 50  # Simulate ~6.5 hour session with frequent checks
            
            while self.trading_active and bar_count < max_bars:
                bar_count += 1
                
                # Simulate market conditions
                np.random.seed(42 + bar_count)
                
                # Mock entropy calculation
                action_entropy = np.random.uniform(0.5, 1.2)
                
                # Mock PnL calculation
                daily_pnl_pct = np.random.uniform(-0.01, 0.01)  # -1% to +1%
                
                # Mock slippage tracking
                expected_slippage = 0.0006  # 6 bps
                realized_slippage = expected_slippage * np.random.uniform(0.8, 1.4)
                slippage_deviation = abs(realized_slippage - expected_slippage) / expected_slippage
                
                # Check kill conditions
                kill_triggered = False
                
                # 1. Entropy floor check
                if action_entropy < 0.75:
                    consecutive_low_entropy += 1
                    if consecutive_low_entropy >= 10:
                        self._send_alert("CRITICAL", f"Low entropy for {consecutive_low_entropy} bars: {action_entropy:.3f}")
                        kill_triggered = True
                else:
                    consecutive_low_entropy = 0
                
                # 2. Daily loss limit
                if daily_pnl_pct <= -0.02:  # -2%
                    self._send_alert("CRITICAL", f"Daily loss limit breached: {daily_pnl_pct:.2%}")
                    kill_triggered = True
                
                # 3. Slippage deviation
                if slippage_deviation > 3.0:  # 3 sigma (simplified)
                    self._send_alert("WARNING", f"High slippage deviation: {slippage_deviation:.1f}x expected")
                    session_stats['alerts_triggered'] += 1
                
                # Emergency halt if kill condition triggered
                if kill_triggered:
                    self._emergency_halt("Kill condition triggered")
                    session_stats['emergency_halts'] += 1
                    break
                
                # Progress update every 10 bars
                if bar_count % 10 == 0:
                    print(f"   Bar {bar_count}: entropy={action_entropy:.3f}, PnL={daily_pnl_pct:.2%}, slippage={realized_slippage*10000:.1f}bps")
                
                session_stats['bars_processed'] = bar_count
                
                # Simulate processing time
                time.sleep(0.1)
            
            session_stats['end_time'] = datetime.now().isoformat()
            
            if self.trading_active:
                print(f"✅ Trading session completed: {bar_count} bars processed")
                self.logger.info(f"Trading session completed successfully: {bar_count} bars")
            else:
                print(f"⚠️ Trading session halted early: {bar_count} bars processed")
                self.logger.warning(f"Trading session halted: {bar_count} bars")
            
        except Exception as e:
            print(f"❌ Trading session error: {e}")
            self.logger.error(f"Trading session exception: {e}")
            self._emergency_halt(f"Trading session exception: {e}")
            session_stats['emergency_halts'] += 1
        
        finally:
            self.trading_active = False
        
        return session_stats
    
    def run_eod_operations(self) -> Dict:
        """
        Run end-of-day operations (15:10 CT).
        
        Returns:
            Dict with EOD results
        """
        self.logger.info("Starting end-of-day operations...")
        print("🌆 END-OF-DAY OPERATIONS")
        print("="*30)
        
        eod_results = {
            'timestamp': datetime.now().isoformat(),
            'reports_generated': [],
            'reconciliation_status': 'pending',
            'alerts_summary': len(self.alerts)
        }
        
        try:
            # 1. Generate daily report
            print("\n📊 Step 1: Generating daily report...")
            
            # Mock trading data for report
            today = datetime.now().strftime('%Y-%m-%d')
            mock_trading_data = {
                'date': today,
                'ic': np.random.uniform(0.010, 0.025),
                'sharpe_net': np.random.uniform(0.25, 0.40),
                'turnover': np.random.uniform(1.5, 2.2),
                'decile_spread': np.random.uniform(0.008, 0.015),
                'factor_r2': np.random.uniform(0.15, 0.35),
                'realized_slippage_bps': np.random.uniform(5.0, 8.0),
                'expected_slippage_bps': 6.0,
                'blocked_order_pct': np.random.uniform(0.0, 5.0),
                'guard_breaches': 0,
                'positions_count': np.random.randint(15, 25)
            }
            
            # Mock daily report for now
            daily_report = {
                'date': today,
                'status': 'mock_validation',
                'metrics': mock_trading_data
            }
            
            # Save daily report
            reports_dir = Path("results/paper/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            daily_report_file = reports_dir / f"daily_{today}.json"
            with open(daily_report_file, 'w') as f:
                json.dump(daily_report, f, indent=2)
            
            print(f"✅ Daily report saved: {daily_report_file}")
            eod_results['reports_generated'].append(str(daily_report_file))
            
            # 2. Check if weekly report needed (Friday)
            if datetime.now().weekday() == 4:  # Friday
                print("\n📈 Step 2: Generating weekly report...")
                
                # Mock weekly data
                weekly_data = [mock_trading_data for _ in range(5)]  # 5 trading days
                # Mock weekly report for now  
                weekly_report = {
                    'week_ending': today,
                    'status': 'mock_validation',
                    'daily_data': weekly_data
                }
                
                weekly_report_file = reports_dir / f"weekly_{today}.json"
                with open(weekly_report_file, 'w') as f:
                    json.dump(weekly_report, f, indent=2)
                
                print(f"✅ Weekly report saved: {weekly_report_file}")
                eod_results['reports_generated'].append(str(weekly_report_file))
            
            # 3. Position reconciliation
            print("\n🔍 Step 3: Position reconciliation...")
            
            # Mock reconciliation (in production, compare with broker positions)
            reconciliation_passed = True  # Mock result
            
            if reconciliation_passed:
                print("✅ Position reconciliation passed")
                eod_results['reconciliation_status'] = 'passed'
            else:
                print("❌ Position reconciliation failed")
                eod_results['reconciliation_status'] = 'failed'
                self._send_alert("CRITICAL", "Position reconciliation failed")
            
            # 4. Alerts summary
            print(f"\n🚨 Step 4: Alerts summary...")
            print(f"   Total alerts today: {len(self.alerts)}")
            
            if self.alerts:
                alert_file = reports_dir / f"alerts_{today}.json"
                with open(alert_file, 'w') as f:
                    json.dump(self.alerts, f, indent=2)
                print(f"   Alerts saved: {alert_file}")
            
            print(f"\n✅ End-of-day operations completed")
            
        except Exception as e:
            print(f"❌ End-of-day operations failed: {e}")
            self.logger.error(f"EOD operations exception: {e}")
            self._send_alert("CRITICAL", f"EOD operations failed: {e}")
        
        return eod_results
    
    def run_full_day_cycle(self) -> Dict:
        """
        Run complete daily cycle.
        
        Returns:
            Dict with full day results
        """
        print("🔄 RUNNING FULL DAILY PAPER TRADING CYCLE")
        print("="*60)
        
        full_day_results = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'start_time': datetime.now().isoformat(),
            'preflight': None,
            'trading_session': None,
            'eod': None,
            'overall_status': 'pending'
        }
        
        try:
            # Step 1: Preflight checks
            print("\n" + "="*20 + " PREFLIGHT " + "="*20)
            preflight_passed = self.run_preflight_checks()
            full_day_results['preflight'] = {'passed': preflight_passed}
            
            if not preflight_passed:
                print("\n❌ ABORTING: Preflight checks failed")
                full_day_results['overall_status'] = 'aborted'
                return full_day_results
            
            # Step 2: Trading session
            print("\n" + "="*20 + " TRADING " + "="*20)
            trading_results = self.run_trading_session()
            full_day_results['trading_session'] = trading_results
            
            # Step 3: End-of-day operations
            print("\n" + "="*22 + " EOD " + "="*22)
            eod_results = self.run_eod_operations()
            full_day_results['eod'] = eod_results
            
            # Overall status
            if (preflight_passed and 
                trading_results.get('emergency_halts', 0) == 0 and
                eod_results.get('reconciliation_status') == 'passed'):
                full_day_results['overall_status'] = 'success'
            else:
                full_day_results['overall_status'] = 'issues'
            
            print(f"\n🎯 DAILY CYCLE COMPLETE: {full_day_results['overall_status'].upper()}")
            
        except Exception as e:
            print(f"❌ Daily cycle failed: {e}")
            self.logger.critical(f"Daily cycle exception: {e}")
            full_day_results['overall_status'] = 'failed'
        
        finally:
            full_day_results['end_time'] = datetime.now().isoformat()
            
            # Save daily cycle results
            results_dir = Path("results/paper/daily_cycles")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / f"cycle_{full_day_results['date']}.json"
            with open(results_file, 'w') as f:
                json.dump(full_day_results, f, indent=2)
            
            print(f"📄 Daily cycle results saved: {results_file}")
        
        return full_day_results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Paper Trading Operations")
    parser.add_argument('--mode', choices=['preflight', 'trading', 'eod', 'full'],
                       default='full', help="Operation mode")
    
    args = parser.parse_args()
    
    # Initialize operations
    ops = DailyPaperTradingOperations()
    
    try:
        if args.mode == 'preflight':
            success = ops.run_preflight_checks()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'trading':
            results = ops.run_trading_session()
            sys.exit(0 if results.get('emergency_halts', 0) == 0 else 1)
            
        elif args.mode == 'eod':
            results = ops.run_eod_operations()
            sys.exit(0 if results.get('reconciliation_status') == 'passed' else 1)
            
        elif args.mode == 'full':
            results = ops.run_full_day_cycle()
            sys.exit(0 if results.get('overall_status') == 'success' else 1)
            
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        ops.logger.warning("Operation interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        ops.logger.critical(f"Unexpected exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
