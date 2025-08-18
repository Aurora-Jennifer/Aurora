#!/usr/bin/env python3
"""
DataSanity Test Runner

Runs the comprehensive DataSanity test suite and provides a one-line summary.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_data_sanity_tests():
    """Run DataSanity tests and return summary."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Change to project root
    os.chdir(project_root)

    # Run pytest with specific options
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_strict_profile.py",
        "tests/test_dtype_casting.py",
        "tests/test_edge_cases.py",
        "tests/test_returns_calc.py",
        "tests/test_corruption_detection.py",
        "tests/test_properties.py",
        "-q",  # Quiet mode
        "--tb=short",  # Short traceback
        "--durations=0",  # No duration summary
        "--disable-warnings",  # Disable warnings
    ]

    try:
        # Run the tests
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        # Parse the output
        output_lines = result.stdout.strip().split("\n")

        # Find the summary line
        summary_line = None
        for line in reversed(output_lines):
            if "passed" in line and "failed" in line:
                summary_line = line
                break

        if summary_line:
            print(f"DataSanity Tests: {summary_line}")
        else:
            print("DataSanity Tests: No summary found")

        # Print any errors
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        # Return exit code
        return result.returncode

    except subprocess.TimeoutExpired:
        print("DataSanity Tests: TIMEOUT - Tests took longer than 5 minutes")
        return 1
    except Exception as e:
        print(f"DataSanity Tests: ERROR - {e}")
        return 1


def main():
    """Main entry point."""
    print("Running DataSanity verification test suite...")
    exit_code = run_data_sanity_tests()

    if exit_code == 0:
        print("✅ All DataSanity tests passed!")
    else:
        print("❌ Some DataSanity tests failed!")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
