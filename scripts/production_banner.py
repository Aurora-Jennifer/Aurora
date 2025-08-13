#!/usr/bin/env python3
"""
Production Readiness Banner
Only shows green when readiness_report.json confirms all checks passed.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def load_readiness_report() -> dict:
    """Load the readiness report."""
    report_path = Path("readiness_report.json")
    if not report_path.exists():
        return {"passed": False, "error": "No readiness report found"}

    try:
        with open(report_path) as f:
            return json.load(f)
    except Exception as e:
        return {"passed": False, "error": f"Failed to load report: {e}"}


def load_falsification_report() -> dict:
    """Load the falsification report."""
    report_path = Path("falsification_report.json")
    if not report_path.exists():
        return {"passed": False, "error": "No falsification report found"}

    try:
        with open(report_path) as f:
            return json.load(f)
    except Exception as e:
        return {"passed": False, "error": f"Failed to load report: {e}"}


def print_production_banner():
    """Print production readiness banner based on actual reports."""
    print("=" * 100)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # Load reports
    readiness_report = load_readiness_report()
    falsification_report = load_falsification_report()

    # Check if both reports exist and passed
    readiness_passed = readiness_report.get("passed", False)
    falsification_passed = falsification_report.get("passed", False)

    if "error" in readiness_report:
        print(f"❌ READINESS REPORT ERROR: {readiness_report['error']}")
        readiness_passed = False

    if "error" in falsification_report:
        print(f"❌ FALSIFICATION REPORT ERROR: {falsification_report['error']}")
        falsification_passed = False

    # Overall status
    all_passed = readiness_passed and falsification_passed

    if all_passed:
        print("\n🎉 ALL SYSTEMS VERIFIED - PRODUCTION READY!")
        print("=" * 100)

        # Show readiness details
        if "summary" in readiness_report:
            summary = readiness_report["summary"]
            print("\n📊 READINESS CHECK RESULTS:")
            print(f"   Pass Rate: {summary.get('pass_rate', 0):.1f}%")
            print(
                f"   Passed: {summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)}"
            )

        # Show falsification details
        if "summary" in falsification_report:
            summary = falsification_report["summary"]
            print("\n🔍 FALSIFICATION TEST RESULTS:")
            print(f"   Pass Rate: {summary.get('pass_rate', 0):.1f}%")
            print(
                f"   Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}"
            )

        print("\n✅ VERIFIED READY FOR AUTONOMOUS PAPER TRADING")
        print("✅ ALL INTEGRITY CHECKS PASSED")
        print("✅ NO LEAKAGE DETECTED")
        print("✅ REALISM KNOBS VERIFIED")
        print("✅ CONSISTENCY CONFIRMED")

    else:
        print("\n❌ PRODUCTION NOT READY - VERIFICATION FAILED")
        print("=" * 100)

        # Show what failed
        if not readiness_passed:
            print("\n❌ READINESS CHECKS FAILED:")
            if "checks" in readiness_report:
                for check in readiness_report["checks"]:
                    if not check.get("passed", False):
                        print(f"   ❌ {check['name']}: {check.get('details', {})}")

        if not falsification_passed:
            print("\n❌ FALSIFICATION TESTS FAILED:")
            if "tests" in falsification_report:
                for test in falsification_report["tests"]:
                    if not test.get("passed", False):
                        print(
                            f"   ❌ {test['test']}: {test.get('error', test.get('reason', 'Unknown error'))}"
                        )

        print("\n🔧 REQUIRED ACTIONS:")
        print("   1. Run: python scripts/readiness_check.py")
        print("   2. Run: python scripts/falsification_tests.py")
        print("   3. Fix all failed checks")
        print("   4. Re-run verification")

    print("\n" + "=" * 100)

    return all_passed


def main():
    """Main function."""
    success = print_production_banner()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
