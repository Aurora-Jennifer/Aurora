#!/usr/bin/env python3
"""
Enhanced DataSanity test runner with fast/slow splits and parallel execution.
"""

import argparse
import os
import subprocess
import sys
import time


def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    print(f"Exit code: {result.returncode}")
    print(f"Time: {elapsed:.2f}s")

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.returncode == 0, elapsed


def run_fast_tests():
    """Run fast tests (exclude slow, network, stress)."""
    cmd = [
        "pytest",
        "-q",
        "-m",
        "not slow and not network and not stress",
        "tests/test_strict_profile.py",
        "tests/test_dtype_casting.py",
        "tests/test_returns_calc.py",
        "tests/test_edge_cases.py",
        "tests/test_corruption_detection.py",
        "tests/sanity/test_cases.py",
        "tests/meta/test_meta_core.py",
        "--tb=short",
    ]
    return run_command(cmd, "Fast Tests (exclude slow/network/stress)")


def run_slow_tests():
    """Run slow tests."""
    cmd = ["pytest", "-q", "-m", "slow", "tests/test_properties.py", "--tb=short"]
    return run_command(cmd, "Slow Tests (property-based)")


def run_all_tests():
    """Run all tests."""
    cmd = [
        "pytest",
        "-q",
        "tests/test_strict_profile.py",
        "tests/test_dtype_casting.py",
        "tests/test_edge_cases.py",
        "tests/test_returns_calc.py",
        "tests/test_corruption_detection.py",
        "tests/test_properties.py",
        "tests/sanity/test_cases.py",
        "tests/meta/test_meta_core.py",
        "--tb=short",
    ]
    return run_command(cmd, "All Tests")


def run_parallel_tests():
    """Run tests in parallel."""
    cmd = [
        "pytest",
        "-q",
        "-n",
        "auto",
        "tests/test_strict_profile.py",
        "tests/test_dtype_casting.py",
        "tests/test_edge_cases.py",
        "tests/test_returns_calc.py",
        "tests/test_corruption_detection.py",
        "tests/sanity/test_cases.py",
        "tests/meta/test_meta_core.py",
        "--tb=short",
    ]
    return run_command(cmd, "Parallel Tests")


def run_coverage():
    """Run tests with coverage."""
    cmd = [
        "pytest",
        "-q",
        "--cov=core.data_sanity",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "tests/test_strict_profile.py",
        "tests/test_dtype_casting.py",
        "tests/test_edge_cases.py",
        "tests/test_returns_calc.py",
        "tests/test_corruption_detection.py",
        "tests/sanity/test_cases.py",
        "tests/meta/test_meta_core.py",
    ]
    return run_command(cmd, "Coverage Report")


def run_performance_tests():
    """Run performance tests."""
    cmd = [
        "pytest",
        "-q",
        "-m",
        "perf or benchmark",
        "-v",
        "--report-log=results/perf.log",
    ]
    return run_command(cmd, "Performance Tests")


def run_stress_tests():
    """Run stress tests."""
    cmd = ["pytest", "-q", "-m", "stress", "tests/test_properties.py", "--tb=short"]
    return run_command(cmd, "Stress Tests")


def run_table_driven_tests():
    """Run table-driven tests."""
    cmd = [
        "pytest",
        "-q",
        "-m",
        "table_driven or yaml",
        "tests/sanity/test_cases.py",
        "--tb=short",
    ]
    return run_command(cmd, "Table-Driven Tests")


def run_metamorphic_tests():
    """Run metamorphic tests."""
    cmd = [
        "pytest",
        "-q",
        "-m",
        "meta or invariance",
        "tests/meta/test_meta_core.py",
        "--tb=short",
    ]
    return run_command(cmd, "Metamorphic Tests")


def run_factory_tests():
    """Run factory-based tests."""
    cmd = ["pytest", "-q", "-m", "factory", "tests/sanity/test_cases.py", "--tb=short"]
    return run_command(cmd, "Factory-Based Tests")


def main():
    parser = argparse.ArgumentParser(description="Enhanced DataSanity Test Runner")
    parser.add_argument(
        "--mode",
        choices=[
            "fast",
            "slow",
            "all",
            "parallel",
            "coverage",
            "perf",
            "stress",
            "table",
            "meta",
            "factory",
        ],
        default="fast",
        help="Test mode to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("🚀 Enhanced DataSanity Test Runner")
    print(f"Mode: {args.mode}")
    print(f"Verbose: {args.verbose}")

    # Set environment variables for testing
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["HYPOTHESIS_PROFILE"] = "default"

    if args.verbose:
        os.environ["DS_VERBOSE_LOGGING"] = "1"

    # Run tests based on mode
    success = False
    elapsed = 0

    if args.mode == "fast":
        success, elapsed = run_fast_tests()
    elif args.mode == "slow":
        success, elapsed = run_slow_tests()
    elif args.mode == "all":
        success, elapsed = run_all_tests()
    elif args.mode == "parallel":
        success, elapsed = run_parallel_tests()
    elif args.mode == "coverage":
        success, elapsed = run_coverage()
    elif args.mode == "perf":
        success, elapsed = run_performance_tests()
    elif args.mode == "stress":
        success, elapsed = run_stress_tests()
    elif args.mode == "table":
        success, elapsed = run_table_driven_tests()
    elif args.mode == "meta":
        success, elapsed = run_metamorphic_tests()
    elif args.mode == "factory":
        success, elapsed = run_factory_tests()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Success: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"Time: {elapsed:.2f}s")

    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
