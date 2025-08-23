#!/usr/bin/env python3
"""
Self-check tool for trading system health.
Runs comprehensive checks without requiring IBKR connection.
"""

import importlib
import json
import os
import sys
from datetime import datetime


def check_python_environment():
    """Check Python environment and dependencies."""
    print("🔍 Checking Python environment...")

    # Check Python version
    python_version = sys.version_info
    print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print("  ❌ Python 3.8+ required")
        return False
    print("  ✅ Python version OK")

    # Check required packages
    required_packages = [
        "pandas",
        "numpy",
        "yfinance",
        "sklearn",
        "matplotlib",
        "seaborn",
        "schedule",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"  Install missing packages: pip install {' '.join(missing_packages)}")
        return False

    return True


def check_file_structure():
    """Check essential files and directories exist."""
    print("\n📁 Checking file structure...")

    required_files = [
        "enhanced_paper_trading.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "CONFIGURATION.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
    ]

    required_dirs = [
        "core",
        "strategies",
        "features",
        "brokers",
        "config",
        "logs",
        "results",
        "data",
        "tests",
    ]

    all_good = True

    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - missing")
            all_good = False

    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ - missing")
            all_good = False

    return all_good


def check_configuration():
    """Check configuration files are valid."""
    print("\n⚙️  Checking configuration...")

    config_files = [
        "config/enhanced_paper_trading_config.json",
        "config/ibkr_config.json",
        "config/strategies_config.json",
    ]

    all_good = True

    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"  ❌ {config_file} - missing")
            all_good = False
            continue

        try:
            with open(config_file) as f:
                json.load(f)
            print(f"  ✅ {config_file} - valid JSON")
        except json.JSONDecodeError as e:
            print(f"  ❌ {config_file} - invalid JSON: {e}")
            all_good = False

    return all_good


def check_logging_directories():
    """Check logging directories exist and are writable."""
    print("\n📝 Checking logging directories...")

    log_dirs = ["logs", "logs/trades", "logs/performance", "logs/errors", "logs/system"]

    all_good = True

    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                print(f"  ✅ {log_dir}/ - created")
            except Exception as e:
                print(f"  ❌ {log_dir}/ - cannot create: {e}")
                all_good = False
        else:
            # Check if writable
            test_file = os.path.join(log_dir, "test_write.tmp")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                print(f"  ✅ {log_dir}/ - writable")
            except Exception as e:
                print(f"  ❌ {log_dir}/ - not writable: {e}")
                all_good = False

    return all_good


def check_imports():
    """Check that all modules can be imported."""
    print("\n📦 Checking module imports...")

    modules_to_test = [
        "core.regime_detector",
        "core.feature_reweighter",
        "core.enhanced_logging",
        "core.notifications",
        "core.utils",
        "strategies.regime_aware_ensemble",
        "strategies.base",
        "strategies.factory",
        "features.feature_engine",
        "features.ensemble",
        "brokers.data_provider",
        "brokers.ibkr_broker",
    ]

    all_good = True

    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - import error: {e}")
            all_good = False

    return all_good


def check_strategy_initialization():
    """Check that strategies can be initialized."""
    print("\n🎯 Checking strategy initialization...")

    try:
        from strategies.regime_aware_ensemble import (
            RegimeAwareEnsembleParams,
            RegimeAwareEnsembleStrategy,
        )

        params = RegimeAwareEnsembleParams()
        RegimeAwareEnsembleStrategy(params)

        print("  ✅ RegimeAwareEnsembleStrategy - initialized")
        return True
    except Exception as e:
        print(f"  ❌ Strategy initialization failed: {e}")
        return False


def check_data_directories():
    """Check data directories exist."""
    print("\n📊 Checking data directories...")

    data_dirs = ["data", "data/ibkr"]

    all_good = True

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"  ✅ {data_dir}/")
        else:
            try:
                os.makedirs(data_dir, exist_ok=True)
                print(f"  ✅ {data_dir}/ - created")
            except Exception as e:
                print(f"  ❌ {data_dir}/ - cannot create: {e}")
                all_good = False

    return all_good


def check_environment_variables():
    """Check environment variables are set."""
    print("\n🌍 Checking environment variables...")

    env_vars = ["IBKR_PAPER_TRADING", "IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID"]


    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var} = {value}")
        else:
            print(f"  ⚠️  {var} - not set (using defaults)")

    return True  # Not critical if not set


def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")

    try:
        import subprocess

        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("  ✅ All tests passed")
            return True
        print(f"  ❌ Tests failed: {result.stderr}")
        return False
    except Exception as e:
        print(f"  ❌ Could not run tests: {e}")
        return False


def main():
    """Run comprehensive self-check."""
    print("🚀 Trading System Self-Check")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    checks = [
        ("Python Environment", check_python_environment),
        ("File Structure", check_file_structure),
        ("Configuration", check_configuration),
        ("Logging Directories", check_logging_directories),
        ("Module Imports", check_imports),
        ("Strategy Initialization", check_strategy_initialization),
        ("Data Directories", check_data_directories),
        ("Environment Variables", check_environment_variables),
        ("Basic Tests", run_tests),
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} - error: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📋 CHECK SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! System is ready.")
        return 0
    print("⚠️  Some checks failed. Review the issues above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
