#!/usr/bin/env python3
"""
Falsification Tests for Trading System
Catch unrealistic performance and verify system integrity.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))


def test_adversarial_replay() -> dict[str, Any]:
    """Test adversarial replay by reversing timestamps."""
    print("Running adversarial replay test...")

    try:
        # Create a copy of SPY data with reversed timestamps
        import yfinance as yf

        # Download original data
        df = yf.download("SPY", start="2024-01-02", end="2024-01-31", auto_adjust=True)

        # Create reversed data
        df_reversed = df.copy()
        df_reversed = df_reversed.iloc[::-1]  # Reverse order
        df_reversed.index = df_reversed.index[::-1]  # Reverse timestamps

        # Save reversed data
        reversed_file = "data/adversarial_spy.csv"
        Path("data").mkdir(exist_ok=True)
        df_reversed.to_csv(reversed_file)

        # Run backtest on reversed data
        test_config = {
            "symbols": ["SPY"],
            "initial_capital": 100000,
            "start_date": "2024-01-02",
            "end_date": "2024-01-31",
            "data_source": "csv",
            "data_file": reversed_file,
        }

        config_path = "config/adversarial_test_config.json"
        with open(config_path, "w") as f:
            json.dump(test_config, f, indent=2)

        cmd = [
            "python",
            "backtest.py",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-31",
            "--config",
            config_path,
            "--symbols",
            "SPY",
        ]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if p.returncode != 0:
            return {
                "test": "adversarial_replay",
                "passed": False,
                "error": f"Backtest failed with exit code {p.returncode}",
                "stderr": p.stderr[-200:] if p.stderr else "No stderr",
            }

        # Parse results
        output = p.stdout

        # Extract metrics
        return_pattern = r"Total Return:\s*([-\d.]+)%"
        sharpe_pattern = r"Sharpe Ratio:\s*([-\d.]+)"

        return_match = re.search(return_pattern, output)
        sharpe_match = re.search(sharpe_pattern, output)

        if return_match and sharpe_match:
            total_return = float(return_match.group(1))
            sharpe = float(sharpe_match.group(1))

            # Check if performance is suspiciously good on reversed data
            # If strategy still performs well on reversed data, it might be leaking
            suspicious = total_return > 5.0 or sharpe > 1.0

            return {
                "test": "adversarial_replay",
                "passed": not suspicious,
                "total_return": total_return,
                "sharpe": sharpe,
                "suspicious": suspicious,
                "reason": (
                    "Good performance on reversed data suggests leakage"
                    if suspicious
                    else "Performance degraded as expected"
                ),
            }
        return {
            "test": "adversarial_replay",
            "passed": False,
            "error": "Could not parse backtest results",
        }

    except Exception as e:
        return {"test": "adversarial_replay", "passed": False, "error": str(e)}


def test_zero_fee_guard() -> dict[str, Any]:
    """Test that zero fees/slippage improves performance."""
    print("Running zero fee guard test...")

    try:
        # Run backtest with realistic fees
        realistic_config = {
            "symbols": ["SPY"],
            "initial_capital": 100000,
            "start_date": "2024-01-02",
            "end_date": "2024-01-31",
            "execution_params": {
                "slippage_bps": 10.0,
                "per_share_fee": 0.005,
                "borrow_rate_annual": 0.05,
            },
        }

        realistic_path = "config/realistic_test_config.json"
        with open(realistic_path, "w") as f:
            json.dump(realistic_config, f, indent=2)

        cmd_realistic = [
            "python",
            "backtest.py",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-31",
            "--config",
            realistic_path,
            "--symbols",
            "SPY",
        ]

        p_realistic = subprocess.run(cmd_realistic, capture_output=True, text=True, timeout=300)

        if p_realistic.returncode != 0:
            return {
                "test": "zero_fee_guard",
                "passed": False,
                "error": f"Realistic backtest failed with exit code {p_realistic.returncode}",
            }

        # Run backtest with zero fees
        zero_fee_config = {
            "symbols": ["SPY"],
            "initial_capital": 100000,
            "start_date": "2024-01-02",
            "end_date": "2024-01-31",
            "execution_params": {
                "slippage_bps": 0.0,
                "per_share_fee": 0.0,
                "borrow_rate_annual": 0.0,
            },
        }

        zero_fee_path = "config/zero_fee_test_config.json"
        with open(zero_fee_path, "w") as f:
            json.dump(zero_fee_config, f, indent=2)

        cmd_zero_fee = [
            "python",
            "backtest.py",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-31",
            "--config",
            zero_fee_path,
            "--symbols",
            "SPY",
        ]

        p_zero_fee = subprocess.run(cmd_zero_fee, capture_output=True, text=True, timeout=300)

        if p_zero_fee.returncode != 0:
            return {
                "test": "zero_fee_guard",
                "passed": False,
                "error": f"Zero fee backtest failed with exit code {p_zero_fee.returncode}",
            }

        # Parse results
        realistic_output = p_realistic.stdout
        zero_fee_output = p_zero_fee.stdout

        # Extract returns
        return_pattern = r"Total Return:\s*([-\d.]+)%"

        realistic_match = re.search(return_pattern, realistic_output)
        zero_fee_match = re.search(return_pattern, zero_fee_output)

        if realistic_match and zero_fee_match:
            realistic_return = float(realistic_match.group(1))
            zero_fee_return = float(zero_fee_match.group(1))

            # Check if zero fees improve performance
            improvement = zero_fee_return - realistic_return

            # Zero fees should improve performance by at least 0.1%
            realistic_improvement = improvement > 0.1

            return {
                "test": "zero_fee_guard",
                "passed": realistic_improvement,
                "realistic_return": realistic_return,
                "zero_fee_return": zero_fee_return,
                "improvement": improvement,
                "realistic_improvement": realistic_improvement,
                "reason": (
                    "Zero fees should improve performance"
                    if realistic_improvement
                    else "Zero fees didn't improve performance - fees may not be wired correctly"
                ),
            }
        return {
            "test": "zero_fee_guard",
            "passed": False,
            "error": "Could not parse backtest results",
        }

    except Exception as e:
        return {"test": "zero_fee_guard", "passed": False, "error": str(e)}


def test_consistency_audit() -> dict[str, Any]:
    """Audit consistency between different runs."""
    print("Running consistency audit...")

    try:
        # Run two identical backtests and compare results
        test_config = {
            "symbols": ["SPY"],
            "initial_capital": 100000,
            "start_date": "2024-01-02",
            "end_date": "2024-01-31",
        }

        config_path = "config/consistency_test_config.json"
        with open(config_path, "w") as f:
            json.dump(test_config, f, indent=2)

        # Run first test
        cmd = [
            "python",
            "backtest.py",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-31",
            "--config",
            config_path,
            "--symbols",
            "SPY",
        ]

        p1 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if p1.returncode != 0:
            return {
                "test": "consistency_audit",
                "passed": False,
                "error": f"First run failed with exit code {p1.returncode}",
            }

        # Run second test
        p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if p2.returncode != 0:
            return {
                "test": "consistency_audit",
                "passed": False,
                "error": f"Second run failed with exit code {p2.returncode}",
            }

        # Parse results
        output1 = p1.stdout
        output2 = p2.stdout

        # Extract metrics
        return_pattern = r"Total Return:\s*([-\d.]+)%"
        sharpe_pattern = r"Sharpe Ratio:\s*([-\d.]+)"

        return1_match = re.search(return_pattern, output1)
        return2_match = re.search(return_pattern, output2)
        sharpe1_match = re.search(sharpe_pattern, output1)
        sharpe2_match = re.search(sharpe_pattern, output2)

        if all([return1_match, return2_match, sharpe1_match, sharpe2_match]):
            return1 = float(return1_match.group(1))
            return2 = float(return2_match.group(1))
            sharpe1 = float(sharpe1_match.group(1))
            sharpe2 = float(sharpe2_match.group(1))

            # Check consistency (should be identical for deterministic runs)
            return_diff = abs(return1 - return2)
            sharpe_diff = abs(sharpe1 - sharpe2)

            consistent = return_diff < 0.01 and sharpe_diff < 0.01  # 1 basis point tolerance

            return {
                "test": "consistency_audit",
                "passed": consistent,
                "run1_return": return1,
                "run2_return": return2,
                "run1_sharpe": sharpe1,
                "run2_sharpe": sharpe2,
                "return_diff": return_diff,
                "sharpe_diff": sharpe_diff,
                "consistent": consistent,
                "reason": (
                    "Results are consistent"
                    if consistent
                    else "Results are inconsistent - system may not be deterministic"
                ),
            }
        return {
            "test": "consistency_audit",
            "passed": False,
            "error": "Could not parse backtest results",
        }

    except Exception as e:
        return {"test": "consistency_audit", "passed": False, "error": str(e)}


def main():
    """Run all falsification tests."""
    print("=" * 80)
    print("FALSIFICATION TESTS")
    print("=" * 80)

    tests = [test_adversarial_replay, test_zero_fee_guard, test_consistency_audit]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{status}: {result['test']}")
            if not result["passed"]:
                print(f"  Error: {result.get('error', result.get('reason', 'Unknown error'))}")
        except Exception as e:
            error_result = {
                "test": test_func.__name__,
                "passed": False,
                "error": str(e),
            }
            results.append(error_result)
            print(f"❌ FAIL: {test_func.__name__} - Exception: {e}")

    # Determine overall status
    passed = all(result["passed"] for result in results)

    # Generate report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed,
        "tests": results,
        "summary": {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r["passed"]),
            "failed_tests": sum(1 for r in results if not r["passed"]),
            "pass_rate": sum(1 for r in results if r["passed"]) / len(results) * 100,
        },
    }

    # Save report
    with open("falsification_report.json", "w") as fp:
        json.dump(report, fp, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("FALSIFICATION TEST SUMMARY")
    print("=" * 80)
    print(
        f"Overall Status: {'✅ SYSTEM INTEGRITY CONFIRMED' if passed else '❌ INTEGRITY ISSUES DETECTED'}"
    )
    print(f"Pass Rate: {report['summary']['pass_rate']:.1f}%")
    print(f"Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")

    if not passed:
        print("\nFailed Tests:")
        for result in results:
            if not result["passed"]:
                print(
                    f"  ❌ {result['test']}: "
                    f"{result.get('error', result.get('reason', 'Unknown error'))}"
                )

    print("\nDetailed report saved to: falsification_report.json")

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    import time

    main()
