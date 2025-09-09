"""
Paper trading operational guards and environment validation.

Prevents accidental live trading and enforces paper-only operation.
"""
import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime
import pandas as pd


class PaperTradingGuards:
    """
    Operational guards to ensure paper trading safety.
    """
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
        
    def enforce_paper_only_mode(self) -> bool:
        """
        Enforce paper trading mode with multiple safeguards.
        
        Returns:
            True if paper mode confirmed, raises exception otherwise
        """
        print("🔒 ENFORCING PAPER TRADING MODE...")
        
        # Check 1: Environment variable
        is_paper_env = os.environ.get('IS_PAPER_TRADING', '').lower() in ['true', '1', 'yes']
        if not is_paper_env:
            raise RuntimeError("CRITICAL: IS_PAPER_TRADING environment variable not set to 'true'")
        
        # Check 2: Live trading keys should not be present
        live_key_vars = [
            'LIVE_API_KEY', 'LIVE_SECRET_KEY', 'LIVE_BROKER_TOKEN',
            'PROD_API_KEY', 'PRODUCTION_KEY', 'REAL_TRADING_KEY'
        ]
        
        for var in live_key_vars:
            if os.environ.get(var):
                raise RuntimeError(f"CRITICAL: Live trading key {var} detected in environment")
        
        # Check 3: Paper broker endpoint validation
        broker_endpoint = os.environ.get('BROKER_ENDPOINT', '')
        paper_endpoints = ['paper', 'sandbox', 'demo', 'test']
        
        if not any(paper_keyword in broker_endpoint.lower() for paper_keyword in paper_endpoints):
            if broker_endpoint:
                raise RuntimeError(f"CRITICAL: Broker endpoint '{broker_endpoint}' does not appear to be paper trading")
            else:
                self.validation_warnings.append("No BROKER_ENDPOINT set - ensure paper broker configuration")
        
        # Check 4: Trading mode configuration file
        config_file = Path("config/trading_mode.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if config.get('mode') != 'paper':
                raise RuntimeError(f"CRITICAL: Trading mode config shows '{config.get('mode')}', expected 'paper'")
        else:
            # Create paper mode config
            config = {
                'mode': 'paper',
                'created_at': datetime.now().isoformat(),
                'live_trading_disabled': True,
                'verification_required': True
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Created paper trading mode config: {config_file}")
        
        print(f"✅ Paper trading mode confirmed and enforced")
        return True
    
    def validate_feature_whitelist_integrity(self, 
                                           whitelist_path: str = "results/production/features_whitelist.json",
                                           expected_hash_path: str = "results/production/features_whitelist.json.hash") -> bool:
        """
        Validate feature whitelist hasn't been tampered with.
        
        Args:
            whitelist_path: Path to feature whitelist
            expected_hash_path: Path to expected hash file
            
        Returns:
            True if integrity check passes
        """
        print("🔐 VALIDATING FEATURE WHITELIST INTEGRITY...")
        
        if not Path(whitelist_path).exists():
            self.validation_errors.append(f"Feature whitelist not found: {whitelist_path}")
            return False
        
        if not Path(expected_hash_path).exists():
            self.validation_errors.append(f"Expected hash not found: {expected_hash_path}")
            return False
        
        # Load whitelist
        with open(whitelist_path, 'r') as f:
            whitelist_features = json.load(f)
        
        # Compute current hash
        content = json.dumps(sorted(whitelist_features), sort_keys=True)
        current_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Load expected hash
        with open(expected_hash_path, 'r') as f:
            expected_hash = f.read().strip()
        
        if current_hash != expected_hash:
            self.validation_errors.append(
                f"Feature whitelist integrity check failed: {current_hash[:12]}... != {expected_hash[:12]}..."
            )
            return False
        
        print(f"✅ Feature whitelist integrity verified: {len(whitelist_features)} features")
        return True
    
    def validate_calendar_configuration(self) -> bool:
        """
        Validate trading calendar and timezone configuration.
        
        Returns:
            True if calendar configuration is valid
        """
        print("📅 VALIDATING CALENDAR CONFIGURATION...")
        
        # Check timezone configuration
        timezone = os.environ.get('TRADING_TIMEZONE', 'America/Chicago')
        if timezone != 'America/Chicago':
            self.validation_warnings.append(f"Trading timezone is {timezone}, expected America/Chicago")
        
        # Test known calendar dates
        try:
            import pandas_market_calendars as mcal
            
            # Get NYSE calendar
            nyse = mcal.get_calendar('NYSE')
            
            # Test a known date range
            schedule = nyse.schedule(start_date='2024-01-01', end_date='2024-01-31')
            
            if len(schedule) == 0:
                self.validation_errors.append("NYSE calendar returned no trading days")
                return False
            
            # Test a known holiday (New Year's Day 2024)
            new_years = pd.Timestamp('2024-01-01', tz='America/New_York')
            is_trading_day = nyse.valid_days(start_date=new_years, end_date=new_years)
            
            if len(is_trading_day) > 0:
                self.validation_warnings.append("Calendar check: 2024-01-01 should be holiday")
            
            print(f"✅ NYSE calendar validated: {len(schedule)} trading days in Jan 2024")
            
        except ImportError:
            self.validation_warnings.append("pandas_market_calendars not available for calendar validation")
        except Exception as e:
            self.validation_warnings.append(f"Calendar validation error: {e}")
        
        return True
    
    def validate_data_governance(self) -> bool:
        """
        Validate data governance requirements.
        
        Returns:
            True if data governance checks pass
        """
        print("📊 VALIDATING DATA GOVERNANCE...")
        
        # Check for sector snapshot
        sector_snapshot_path = Path("snapshots/sector_map.parquet")
        if not sector_snapshot_path.exists():
            self.validation_warnings.append(f"Sector snapshot not found: {sector_snapshot_path}")
        else:
            print(f"✅ Sector snapshot found: {sector_snapshot_path}")
        
        # Check data freshness requirements
        max_data_age_hours = 24
        current_time = datetime.now()
        
        # In a real system, check actual data timestamps
        data_paths = [
            "data/latest/prices.parquet",
            "data/latest/fundamentals.parquet"
        ]
        
        for data_path in data_paths:
            if Path(data_path).exists():
                # Would check actual file modification time in production
                print(f"✅ Data path exists: {data_path}")
            else:
                self.validation_warnings.append(f"Expected data path not found: {data_path}")
        
        return True
    
    def validate_risk_controls(self) -> bool:
        """
        Validate risk control configuration.
        
        Returns:
            True if risk controls are properly configured
        """
        print("🛡️ VALIDATING RISK CONTROLS...")
        
        # Check risk limits configuration
        risk_config_path = Path("config/risk_limits.json")
        
        default_risk_config = {
            'max_gross_exposure_pct': 30.0,
            'max_position_pct': 2.0,
            'max_adv_participation_pct': 2.0,
            'daily_loss_kill_pct': 2.0,
            'min_price_dollars': 5.0,
            'max_trades_per_day': 60,
            'enable_short_selling': False  # Disable for paper trading
        }
        
        if not risk_config_path.exists():
            # Create default risk config
            with open(risk_config_path, 'w') as f:
                json.dump(default_risk_config, f, indent=2)
            print(f"✅ Created default risk config: {risk_config_path}")
        else:
            with open(risk_config_path, 'r') as f:
                risk_config = json.load(f)
            
            # Validate critical limits
            if risk_config.get('daily_loss_kill_pct', 0) > 5.0:
                self.validation_warnings.append("Daily loss kill limit > 5% seems high")
            
            if risk_config.get('max_position_pct', 0) > 5.0:
                self.validation_warnings.append("Max position limit > 5% seems high")
            
            print(f"✅ Risk controls configuration validated")
        
        return True
    
    def validate_secrets_management(self) -> bool:
        """
        Validate secrets management practices.
        
        Returns:
            True if secrets are properly managed
        """
        print("🔑 VALIDATING SECRETS MANAGEMENT...")
        
        # Check for exposed secrets in config files
        config_files = list(Path("config").glob("**/*.yaml")) + list(Path("config").glob("**/*.json"))
        
        secret_patterns = [
            r'password\s*[:=]\s*["\']?[^"\'\s]+',
            r'api[_-]?key\s*[:=]\s*["\']?[^"\'\s]+',
            r'secret\s*[:=]\s*["\']?[^"\'\s]+',
            r'token\s*[:=]\s*["\']?[^"\'\s]+'
        ]
        
        import re
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.validation_warnings.append(f"Potential secret in config file: {config_file}")
                        break
                        
            except Exception:
                continue  # Skip files that can't be read
        
        # Check required environment variables are present
        required_env_vars = ['IS_PAPER_TRADING']
        
        for var in required_env_vars:
            if not os.environ.get(var):
                self.validation_errors.append(f"Required environment variable not set: {var}")
        
        print(f"✅ Secrets management validation completed")
        return True
    
    def run_comprehensive_validation(self) -> Dict:
        """
        Run comprehensive pre-trading validation.
        
        Returns:
            Dict with validation results
        """
        print("🔍 COMPREHENSIVE PRE-TRADING VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        # Run all validation checks
        checks = [
            ('paper_mode', self.enforce_paper_only_mode),
            ('whitelist_integrity', self.validate_feature_whitelist_integrity),
            ('calendar_config', self.validate_calendar_configuration),
            ('data_governance', self.validate_data_governance),
            ('risk_controls', self.validate_risk_controls),
            ('secrets_management', self.validate_secrets_management)
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                validation_results[check_name] = {
                    'passed': result,
                    'errors': [],
                    'warnings': []
                }
            except Exception as e:
                validation_results[check_name] = {
                    'passed': False,
                    'errors': [str(e)],
                    'warnings': []
                }
                self.validation_errors.append(f"{check_name}: {e}")
        
        # Summary
        total_errors = len(self.validation_errors)
        total_warnings = len(self.validation_warnings)
        all_passed = total_errors == 0
        
        summary = {
            'validation_passed': all_passed,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'errors': self.validation_errors,
            'warnings': self.validation_warnings,
            'check_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save validation report
        report_path = Path("results/validation/pre_trading_validation.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Total errors: {total_errors}")
        print(f"   Total warnings: {total_warnings}")
        print(f"   Status: {'✅ READY FOR PAPER TRADING' if all_passed else '❌ ISSUES MUST BE RESOLVED'}")
        
        if self.validation_errors:
            print(f"\n❌ ERRORS (must fix):")
            for error in self.validation_errors:
                print(f"   - {error}")
        
        if self.validation_warnings:
            print(f"\n⚠️ WARNINGS (review recommended):")
            for warning in self.validation_warnings:
                print(f"   - {warning}")
        
        if all_passed:
            print(f"\n🚀 PAPER TRADING CLEARED FOR LAUNCH!")
        else:
            print(f"\n🛑 RESOLVE ERRORS BEFORE STARTING PAPER TRADING")
        
        return summary


def set_paper_trading_environment():
    """Set up paper trading environment variables."""
    print("🔧 CONFIGURING PAPER TRADING ENVIRONMENT...")
    
    # Set critical environment variables
    os.environ['IS_PAPER_TRADING'] = 'true'
    os.environ['TRADING_TIMEZONE'] = 'America/Chicago'
    os.environ['BROKER_ENDPOINT'] = 'https://paper-api.alpaca.markets'  # Example
    os.environ['PYTHONHASHSEED'] = '42'  # For reproducibility
    
    print(f"✅ Paper trading environment configured")


def main():
    """CLI entry point for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading Guards and Validation")
    parser.add_argument('--setup-env', action='store_true',
                       help="Set up paper trading environment")
    parser.add_argument('--validate', action='store_true',
                       help="Run comprehensive validation")
    
    args = parser.parse_args()
    
    if args.setup_env:
        set_paper_trading_environment()
    
    if args.validate or not (args.setup_env):
        # Default to validation if no specific action
        guards = PaperTradingGuards()
        result = guards.run_comprehensive_validation()
        
        # Exit with error code if validation failed
        sys.exit(0 if result['validation_passed'] else 1)


if __name__ == "__main__":
    main()
