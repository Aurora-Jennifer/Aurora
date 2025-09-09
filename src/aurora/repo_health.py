#!/usr/bin/env python3
"""
Repository Health Check - Comprehensive validation suite

This script runs the complete validation suite to ensure:
1. No shadow decision paths bypass the unified decision core
2. Static analysis passes (types, style, dead code, deps)
3. Contract tests validate decision core behavior
4. Determinism and falsification tests pass
5. Runtime invariants are enforced

Exit codes:
- 0: All checks pass
- 1: Any check fails
"""

import argparse
import subprocess
import sys
from pathlib import Path


class RepoHealthChecker:
    """Comprehensive repository health checker"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.failures = []
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with level"""
        if level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️  {message}")
        elif level == "SUCCESS":
            print(f"✅ {message}")
        else:
            print(f"ℹ️  {message}")
    
    def run_command(self, cmd: list[str], description: str) -> tuple[bool, str]:
        """Run command and return success status and output"""
        try:
            if self.verbose:
                self.log(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                self.log(f"{description} - PASSED", "SUCCESS")
                return True, result.stdout
            self.log(f"{description} - FAILED", "ERROR")
            if result.stderr:
                self.log(f"Error: {result.stderr}", "ERROR")
            return False, result.stderr
                
        except Exception as e:
            self.log(f"{description} - ERROR: {e}", "ERROR")
            return False, str(e)
    
    def check_static_analysis(self) -> bool:
        """Run static analysis suite"""
        self.log("🔍 Running Static Analysis Suite...")
        
        checks = [
            (["ruff", "check", "."], "Ruff linting"),
            (["ruff", "format", "--check", "."], "Ruff formatting"),
            (["mypy", "--strict", "core/", "scripts/"], "MyPy type checking"),
            (["vulture", "core/", "scripts/", "--min-confidence", "80"], "Dead code detection"),
            (["deptry", "."], "Dependency analysis"),
            (["bandit", "-q", "-r", "core/", "scripts/"], "Security scan"),
            (["pip-audit"], "Vulnerability scan")
        ]
        
        all_passed = True
        for cmd, description in checks:
            success, _ = self.run_command(cmd, description)
            if not success:
                all_passed = False
                self.failures.append(f"Static analysis failed: {description}")
        
        return all_passed
    
    def check_shadow_decision_paths(self) -> bool:
        """Check for shadow decision paths using grep"""
        self.log("🔍 Checking for Shadow Decision Paths...")
        
        # Look for decision logic outside decision_core.py
        patterns = [
            (r"(BUY|SELL|HOLD)", "Action constants outside decision core"),
            (r"action\s*=\s*[012]", "Action assignments outside decision core"),
            (r"position\s*=", "Position assignments outside decision core"),
            (r"tau\s*[<>=]", "Tau usage outside decision core")
        ]
        
        all_passed = True
        for pattern, description in patterns:
            try:
                # Use ripgrep to find patterns
                result = subprocess.run(
                    ["rg", "-n", pattern, "core/", "scripts/"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    # Filter out decision_core.py and test files
                    filtered_lines = [
                        line for line in lines 
                        if 'decision_core.py' not in line and 'test_' not in line
                    ]
                    
                    if filtered_lines:
                        self.log(f"Found {len(filtered_lines)} instances of {description}", "ERROR")
                        for line in filtered_lines[:5]:  # Show first 5
                            self.log(f"  {line}", "ERROR")
                        if len(filtered_lines) > 5:
                            self.log(f"  ... and {len(filtered_lines) - 5} more", "ERROR")
                        all_passed = False
                        self.failures.append(f"Shadow decision paths: {description}")
                    else:
                        self.log(f"No shadow {description.lower()} found", "SUCCESS")
                else:
                    self.log(f"No instances of {description} found", "SUCCESS")
                    
            except FileNotFoundError:
                self.log("ripgrep not found, skipping shadow path check", "WARNING")
                self.warnings.append("ripgrep not available for shadow path detection")
                break
        
        return all_passed
    
    def check_contract_tests(self) -> bool:
        """Run contract tests for decision core"""
        self.log("🔍 Running Decision Core Contract Tests...")
        
        # Check if contract tests exist
        contract_test_file = Path("tests/test_decision_contract.py")
        if not contract_test_file.exists():
            self.log("Contract tests not found, creating them...", "WARNING")
            self.create_contract_tests()
        
        # Run contract tests
        success, output = self.run_command(
            ["python", "-m", "pytest", "tests/test_decision_contract.py", "-v"],
            "Decision core contract tests"
        )
        
        if not success:
            self.failures.append("Contract tests failed")
        
        return success
    
    def create_contract_tests(self):
        """Create property-based contract tests"""
        contract_test_content = '''#!/usr/bin/env python3
"""
Property-based contract tests for decision core
"""

import math
import pytest
from hypothesis import given, strategies as st
import sys
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.decision_core import apply_decision, BUY, SELL, HOLD


@given(
    edge=st.floats(allow_nan=False, allow_infinity=False, width=32),
    tau=st.floats(min_value=0.0, max_value=1e-2),
    prev=st.integers(min_value=-1, max_value=1),
    comm=st.floats(min_value=0.0, max_value=50.0),
    slip=st.floats(min_value=0.0, max_value=50.0)
)
def test_decision_core_contract(edge, tau, prev, comm, slip):
    """Test decision core contracts with property-based testing"""
    costs = {"commission_bps": comm, "slippage_bps": slip}
    
    act, pos, cost = apply_decision(edge, tau, prev, costs, fail_fast=False)
    
    # Action is one of {BUY, SELL, HOLD}
    assert act in (BUY, SELL, HOLD)
    
    # HOLD preserves position; non-HOLD sets pos==act
    if act == HOLD:
        assert pos == prev
    else:
        assert pos == act
    
    # Costs only on position change; otherwise 0
    expected_cost = 0.0 if pos == prev else (comm + slip) / 10000.0
    assert math.isclose(cost, expected_cost, rel_tol=0, abs_tol=1e-9)
    
    # Threshold rule: only act when |edge|>=tau
    if abs(edge) < tau:
        assert act == HOLD


def test_decision_core_nan_handling():
    """Test NaN/Inf handling"""
    costs = {"commission_bps": 1.0, "slippage_bps": 3.0}
    
    # Test NaN
    act, pos, cost = apply_decision(float('nan'), 0.001, 0, costs, fail_fast=False)
    assert act == HOLD
    assert pos == 0
    assert cost == 0.0
    
    # Test Inf
    act, pos, cost = apply_decision(float('inf'), 0.001, 0, costs, fail_fast=False)
    assert act == HOLD
    assert pos == 0
    assert cost == 0.0


def test_decision_core_fail_fast():
    """Test fail_fast behavior"""
    costs = {"commission_bps": 1.0, "slippage_bps": 3.0}
    
    # Should raise on NaN with fail_fast=True
    with pytest.raises(ValueError, match="NaN/Inf edge detected"):
        apply_decision(float('nan'), 0.001, 0, costs, fail_fast=True)
    
    # Should raise on Inf with fail_fast=True
    with pytest.raises(ValueError, match="NaN/Inf edge detected"):
        apply_decision(float('inf'), 0.001, 0, costs, fail_fast=True)
'''
        
        # Create tests directory if it doesn't exist
        tests_dir = Path("tests")
        tests_dir.mkdir(exist_ok=True)
        
        # Write contract tests
        with open(contract_test_file, 'w') as f:
            f.write(contract_test_content)
        
        self.log("Created contract tests", "SUCCESS")
    
    def check_determinism_tests(self) -> bool:
        """Run determinism and falsification tests"""
        self.log("🔍 Running Determinism and Falsification Tests...")
        
        # Check if falsification harness exists
        falsification_script = Path("scripts/falsification_harness.py")
        if not falsification_script.exists():
            self.log("Falsification harness not found", "WARNING")
            self.warnings.append("Falsification harness not available")
            return True
        
        # Run a quick determinism test
        success, output = self.run_command(
            ["python", "scripts/falsification_harness.py", "--config", "configs/paper.yaml", "--quick"],
            "Determinism test"
        )
        
        if not success:
            self.failures.append("Determinism test failed")
        
        return success
    
    def check_runtime_invariants(self) -> bool:
        """Test runtime invariant enforcement"""
        self.log("🔍 Testing Runtime Invariants...")
        
        # Test config validation
        test_config = {
            "runtime": {
                "single_decision_core": True,
                "allow_legacy_paths": False,
                "fail_fast_on_nan": True
            },
            "decision": {
                "tau_threshold": 0.0001
            }
        }
        
        try:
            from core.decision_core import validate_system_startup
            validate_system_startup(test_config)
            self.log("Runtime invariants validation - PASSED", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"Runtime invariants validation - FAILED: {e}", "ERROR")
            self.failures.append(f"Runtime invariants failed: {e}")
            return False
    
    def check_coverage(self) -> bool:
        """Check test coverage"""
        self.log("🔍 Checking Test Coverage...")
        
        # Run tests with coverage
        success, output = self.run_command(
            ["python", "-m", "pytest", "--cov=core", "--cov=scripts", "--cov-report=term-missing:skip-covered", "--cov-report=xml"],
            "Test coverage"
        )
        
        if not success:
            self.failures.append("Coverage test failed")
            return False
        
        # Check coverage threshold
        try:
            import xml.etree.ElementTree as ET
            if Path("coverage.xml").exists():
                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                line_rate = float(root.attrib.get('line-rate', 0))
                coverage_pct = line_rate * 100
                
                if coverage_pct >= 80:  # 80% threshold
                    self.log(f"Coverage {coverage_pct:.1f}% - PASSED", "SUCCESS")
                    return True
                self.log(f"Coverage {coverage_pct:.1f}% < 80% - FAILED", "ERROR")
                self.failures.append(f"Coverage too low: {coverage_pct:.1f}%")
                return False
            self.log("Coverage XML not found", "WARNING")
            return True
        except Exception as e:
            self.log(f"Coverage check failed: {e}", "WARNING")
            return True
    
    def run_all_checks(self) -> bool:
        """Run all health checks"""
        self.log("🚀 Starting Repository Health Check...")
        self.log("=" * 50)
        
        checks = [
            self.check_static_analysis,
            self.check_shadow_decision_paths,
            self.check_contract_tests,
            self.check_determinism_tests,
            self.check_runtime_invariants,
            self.check_coverage
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.log(f"Check failed with exception: {e}", "ERROR")
                self.failures.append(f"Check exception: {e}")
                all_passed = False
        
        # Summary
        self.log("=" * 50)
        if all_passed:
            self.log("🎉 All health checks PASSED!", "SUCCESS")
            if self.warnings:
                self.log(f"⚠️  {len(self.warnings)} warnings (non-blocking)", "WARNING")
        else:
            self.log(f"❌ {len(self.failures)} health checks FAILED!", "ERROR")
            for failure in self.failures:
                self.log(f"  - {failure}", "ERROR")
        
        return all_passed


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Repository Health Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick check (skip slow tests)")
    
    args = parser.parse_args()
    
    checker = RepoHealthChecker(verbose=args.verbose)
    
    if args.quick:
        # Quick check - just static analysis and shadow paths
        checker.log("🚀 Running Quick Health Check...")
        success = (
            checker.check_static_analysis() and
            checker.check_shadow_decision_paths()
        )
    else:
        # Full check
        success = checker.run_all_checks()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
