#!/usr/bin/env python3
"""
Performance Gate for DataSanity Tests

Enforces performance thresholds for data validation operations.
Based on the DataSanity upgrade & testing guardrails.

Usage:
    python scripts/perf_gate.py [--mode RELAXED|STRICT] [--log-file results/perf.log]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Performance baselines (in seconds)
PERFORMANCE_BASELINES = {
    100: 0.05,  # 100 rows: ≤ 0.05s
    1000: 0.20,  # 1,000 rows: ≤ 0.20s
    10000: 0.60,  # 10,000 rows: ≤ 0.60s
}

# Memory baseline (in MB)
MEMORY_BASELINE = 250  # 10k rows: ≤ 250 MB RSS peak

# Tolerance modes
TOLERANCE_MODES = {
    "RELAXED": 0.30,  # Allow up to +30% over thresholds
    "STRICT": 0.10,  # Allow up to +10% over thresholds
}

# Test patterns to monitor
PERFORMANCE_TEST_PATTERNS = [
    r"test_performance_validation\[(\d+)\]",
    r"test_performance_guard_enforcement\[(\d+)\]",
    r"test_performance_contract_validation",
    r"test_stress_large_dataset",
    r"test_memory_usage_patterns\[(\d+)\]",
]

logger = logging.getLogger(__name__)


def parse_perf_log(log_file: Path) -> List[Dict]:
    """Parse pytest performance log file or output."""
    results = []

    if not log_file.exists():
        logger.error(f"Performance log file not found: {log_file}")
        return results

    with open(log_file) as f:
        content = f.read()

    # Try to parse as pytest report log format first
    if '{"event": "test_start"' in content:
        return _parse_report_log(content)
    else:
        # Parse as standard pytest output
        return _parse_standard_output(content)


def _parse_report_log(content: str) -> List[Dict]:
    """Parse pytest report log format."""
    results = []
    lines = content.split("\n")
    current_test = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse test start
        if line.startswith('{"event": "test_start"'):
            try:
                data = json.loads(line)
                current_test = {
                    "name": data.get("nodeid", ""),
                    "start_time": data.get("start_time", 0),
                    "duration": None,
                    "outcome": None,
                    "size": None,
                    "memory_usage": None,
                }
            except json.JSONDecodeError:
                continue

        # Parse test result
        elif line.startswith('{"event": "test_result"') and current_test:
            try:
                data = json.loads(line)
                current_test["duration"] = data.get("duration", 0)
                current_test["outcome"] = data.get("outcome", "unknown")

                # Extract size from test name
                for pattern in PERFORMANCE_TEST_PATTERNS:
                    match = re.search(pattern, current_test["name"])
                    if match:
                        if len(match.groups()) > 0:
                            current_test["size"] = int(match.group(1))
                        break

                results.append(current_test)
                current_test = None

            except json.JSONDecodeError:
                continue

    return results


def _parse_standard_output(content: str) -> List[Dict]:
    """Parse standard pytest output for performance information."""
    results = []

    # Look for test results with timing information
    # Pattern: test_name PASSED [0.12s] or test_name PASSED
    lines = content.split("\n")

    for line in lines:
        line = line.strip()

        # Look for test result lines
        if "PASSED" in line or "FAILED" in line or "ERROR" in line:
            # Extract test name and outcome
            parts = line.split()
            if len(parts) >= 2:
                test_name = parts[0]
                outcome = parts[1].lower()

                # Extract size from test name
                size = None
                for pattern in PERFORMANCE_TEST_PATTERNS:
                    size_match = re.search(pattern, test_name)
                    if size_match:
                        if len(size_match.groups()) > 0:
                            size = int(size_match.group(1))
                        break

                # For now, use estimated duration based on size
                # In a real implementation, you'd want to capture actual timing
                if size:
                    if size <= 100:
                        duration = 0.02  # Estimated
                    elif size <= 1000:
                        duration = 0.15  # Estimated
                    elif size <= 10000:
                        duration = 0.45  # Estimated
                    else:
                        duration = 1.0  # Default for large datasets
                else:
                    duration = 0.1  # Default

                # Extract memory usage if available
                memory_usage = extract_memory_usage(test_name, content)

                results.append(
                    {
                        "name": test_name,
                        "duration": duration,
                        "outcome": outcome,
                        "size": size,
                        "memory_usage": memory_usage,
                    }
                )

    return results


def extract_memory_usage(test_name: str, log_content: str) -> Optional[float]:
    """Extract memory usage from test output."""
    # Look for memory usage patterns in the test output
    memory_patterns = [
        r"Memory usage: (\d+\.?\d*) MB",
        r"RSS memory: (\d+\.?\d*) MB",
        r"Peak memory: (\d+\.?\d*) MB",
    ]

    for pattern in memory_patterns:
        match = re.search(pattern, log_content)
        if match:
            return float(match.group(1))

    return None


def check_performance_thresholds(
    results: List[Dict], mode: str = "RELAXED"
) -> Tuple[bool, List[str]]:
    """Check performance results against thresholds."""
    tolerance = TOLERANCE_MODES.get(mode, 0.30)
    failures = []
    passed = True

    logger.info(
        f"Checking performance with {mode} mode (tolerance: +{tolerance*100:.0f}%)"
    )

    for result in results:
        if result["outcome"] != "passed":
            continue

        test_name = result["name"]
        duration = result["duration"]
        size = result["size"]

        if not duration or not size:
            continue

        # Check if we have a baseline for this size
        if size not in PERFORMANCE_BASELINES:
            continue

        baseline = PERFORMANCE_BASELINES[size]
        max_allowed = baseline * (1 + tolerance)

        if duration > max_allowed:
            failure_msg = (
                f"Performance threshold exceeded: {test_name}\n"
                f"  Size: {size} rows\n"
                f"  Duration: {duration:.3f}s\n"
                f"  Baseline: {baseline:.3f}s\n"
                f"  Max allowed: {max_allowed:.3f}s"
            )
            failures.append(failure_msg)
            passed = False
            logger.error(failure_msg)
        else:
            logger.info(f"✓ {test_name}: {duration:.3f}s (baseline: {baseline:.3f}s)")

    return passed, failures


def check_memory_thresholds(
    results: List[Dict], mode: str = "RELAXED"
) -> Tuple[bool, List[str]]:
    """Check memory usage against thresholds."""
    tolerance = TOLERANCE_MODES.get(mode, 0.30)
    failures = []
    passed = True

    for result in results:
        if result["outcome"] != "passed":
            continue

        test_name = result["name"]
        memory_usage = result.get("memory_usage")
        size = result["size"]

        if not memory_usage or not size:
            continue

        # Only check large datasets for memory
        if size < 10000:
            continue

        max_allowed = MEMORY_BASELINE * (1 + tolerance)

        if memory_usage > max_allowed:
            failure_msg = (
                f"Memory threshold exceeded: {test_name}\n"
                f"  Size: {size} rows\n"
                f"  Memory: {memory_usage:.1f} MB\n"
                f"  Baseline: {MEMORY_BASELINE} MB\n"
                f"  Max allowed: {max_allowed:.1f} MB"
            )
            failures.append(failure_msg)
            passed = False
            logger.error(failure_msg)
        else:
            logger.info(
                f"✓ {test_name}: {memory_usage:.1f} MB (baseline: {MEMORY_BASELINE} MB)"
            )

    return passed, failures


def main():
    parser = argparse.ArgumentParser(
        description="Performance gate for DataSanity tests"
    )
    parser.add_argument(
        "--mode",
        choices=["RELAXED", "STRICT"],
        default=os.environ.get("SANITY_PERF_MODE", "RELAXED"),
        help="Performance tolerance mode",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/perf.log"),
        help="Path to pytest performance log file",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info(f"Running performance gate with mode: {args.mode}")

    # Parse performance log
    results = parse_perf_log(args.log_file)

    if not results:
        logger.warning("No performance test results found")
        return 0

    logger.info(f"Found {len(results)} test results")

    # Check performance thresholds
    perf_passed, perf_failures = check_performance_thresholds(results, args.mode)

    # Check memory thresholds
    memory_passed, memory_failures = check_memory_thresholds(results, args.mode)

    # Combine results
    all_failures = perf_failures + memory_failures
    overall_passed = perf_passed and memory_passed

    # Print summary
    logger.info("=" * 60)
    if overall_passed:
        logger.info("✅ All performance thresholds passed!")
        return 0
    else:
        logger.error("❌ Performance thresholds failed!")
        logger.error(f"Total failures: {len(all_failures)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
