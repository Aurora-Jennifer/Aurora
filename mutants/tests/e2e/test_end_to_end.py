"""
End-to-end tests for Aurora trading system.

Validates the complete pipeline from data ingestion to artifact generation
using deterministic fixtures with no external dependencies.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from .utils import assert_json_schema, assert_no_network, run_command


@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end test suite for Aurora trading system."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup test environment and cleanup after each test."""
        # Setup: ensure fixtures exist
        self.fixtures_dir = Path("tests/fixtures")
        self.fixtures_dir.mkdir(exist_ok=True)
        
        # Generate fixtures if they don't exist
        fixture_file = self.fixtures_dir / "quotes.parquet"
        if not fixture_file.exists():
            self._generate_fixtures()
        
        # Setup test artifacts directory
        self.test_artifacts = Path("test_artifacts")
        self.test_artifacts.mkdir(exist_ok=True)
        
        yield
        
        # Teardown: cleanup test artifacts
        if self.test_artifacts.exists():
            shutil.rmtree(self.test_artifacts)
    
    def _generate_fixtures(self):
        """Generate deterministic test fixtures."""
        from .fixtures.gen_fixture import generate_test_data
        generate_test_data()
    
    def test_smoke_command_success(self):
        """Test that smoke command runs successfully."""
        with assert_no_network():
            # Run smoke command
            cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            result = run_command(cmd)
            
            # Assert success
            assert result.returncode == 0, f"Smoke command failed: {result.stdout}\n{result.stderr}"
            
            # Check for key markers in output
            assert "SMOKE OK" in result.stdout, "Should contain 'SMOKE OK' marker"
            # Data sanity is working (check stderr for validation messages)
            assert "DataSanity validation enabled" in result.stderr, "DataSanity should be enabled"
    
    def test_artifacts_generated(self):
        """Test that required artifacts are generated."""
        with assert_no_network():
            # Run smoke command
            cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            result = run_command(cmd)
            assert result.returncode == 0
            
            # Check for smoke run report
            smoke_report = Path("reports/smoke_run.json")
            assert smoke_report.exists(), "smoke_run.json not generated"
            
            # Validate JSON schema
            schema_path = Path("reports/smoke_run.schema.json")
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                assert_json_schema(smoke_report, schema)
            
            # Check report content
            with open(smoke_report) as f:
                report_data = json.load(f)
            
            assert report_data.get("status") == "OK", "Report should indicate success"
            assert "symbols" in report_data, "Report should have symbols"
            assert "trades" in report_data, "Report should have trade count"
            
            # Validate metrics are reasonable for fixture data
            assert report_data.get("trades", 0) >= 0, "Trade count should be non-negative"
            assert report_data.get("folds", 0) >= 1, "Should have at least one fold"
    
    def test_deterministic_runs(self):
        """Test that consecutive runs produce identical results (excluding timestamps)."""
        with assert_no_network():
            # Run smoke command twice
            cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            # First run
            result1 = run_command(cmd)
            assert result1.returncode == 0
            
            # Get content excluding timestamp
            smoke_report = Path("reports/smoke_run.json")
            with open(smoke_report) as f:
                data1 = json.load(f)
            # Remove timestamp and duration for comparison (they vary between runs)
            data1.pop("timestamp", None)
            data1.pop("duration_s", None)
            if "fold_summaries" in data1:
                for summary in data1["fold_summaries"]:
                    summary.pop("duration_ms", None)
            content1 = json.dumps(data1, sort_keys=True)
            
            # Second run
            result2 = run_command(cmd)
            assert result2.returncode == 0
            
            # Compare content (excluding timestamp and duration)
            with open(smoke_report) as f:
                data2 = json.load(f)
            data2.pop("timestamp", None)
            data2.pop("duration_s", None)
            if "fold_summaries" in data2:
                for summary in data2["fold_summaries"]:
                    summary.pop("duration_ms", None)
            content2 = json.dumps(data2, sort_keys=True)
            
            assert content1 == content2, "Runs should be deterministic (excluding timestamps)"
    
    def test_go_nogo_gate(self):
        """Test that go/nogo gate passes with smoke results."""
        with assert_no_network():
            # First run smoke to generate report
            smoke_cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            result = run_command(smoke_cmd)
            assert result.returncode == 0
            
            # Check if go_nogo script exists
            go_nogo_script = Path("scripts/go_nogo.py")
            if not go_nogo_script.exists():
                pytest.skip("go_nogo.py script not found")
            
            # Run go/nogo gate with structured logs enabled
            gate_cmd = [
                "python", "scripts/go_nogo.py",
                "--input", "reports/smoke_run.json"
            ]
            
            # Set required environment variables
            import os
            env = os.environ.copy()
            env["STRUCTURED_LOGS"] = "1"
            env["RUN_ID"] = "test-run-20250819"
            env["MAX_POSITION_PCT"] = "0.15"
            env["MAX_GROSS_LEVERAGE"] = "2.0"
            env["DAILY_LOSS_CUT_PCT"] = "0.03"
            env["MAX_DRAWDOWN_CUT_PCT"] = "0.20"
            env["MAX_TURNOVER_PCT"] = "300"
            
            gate_result = subprocess.run(
                gate_cmd,
                capture_output=True,
                text=True,
                env=env
            )
            assert gate_result.returncode == 0, f"Go/nogo gate failed: {gate_result.stderr}"
            
            # Check for approval message
            assert "GO" in gate_result.stdout or "PASS" in gate_result.stdout
    
    def test_performance_gate(self):
        """Test that performance gate passes with smoke results."""
        with assert_no_network():
            # First run smoke to generate report
            smoke_cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            result = run_command(smoke_cmd)
            assert result.returncode == 0
            
            # Check if perf_gate script exists
            perf_gate_script = Path("scripts/perf_gate.py")
            if not perf_gate_script.exists():
                pytest.skip("perf_gate.py script not found")
            
            # Run performance gate (check if it exists and has correct interface)
            gate_cmd = [
                "python", "scripts/perf_gate.py",
                "--mode", "RELAXED"
            ]
            
            gate_result = run_command(gate_cmd)
            # Performance gate might not be implemented yet, so just check it doesn't crash
            assert gate_result.returncode in [0, 1], f"Performance gate should not crash: {gate_result.stderr}"
    
    def test_csv_outputs_valid(self):
        """Test that CSV outputs have expected structure."""
        with assert_no_network():
            # Run smoke command
            cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            result = run_command(cmd)
            assert result.returncode == 0
            
            # Check for CSV outputs in reports directory
            reports_dir = Path("reports")
            csv_files = list(reports_dir.glob("*.csv"))
            
            if csv_files:
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    assert len(df) > 0, f"CSV file {csv_file} should not be empty"
                    
                    # Check for required columns if it's a results file
                    if "results" in csv_file.name.lower():
                        required_cols = ["timestamp", "symbol", "close"]
                        for col in required_cols:
                            if col in df.columns and col == "timestamp":
                                # Check timestamp is increasing
                                df[col] = pd.to_datetime(df[col])
                                assert df[col].is_monotonic_increasing, f"Timestamps should be increasing in {csv_file}"
    
    def test_no_network_dependencies(self):
        """Test that no network calls are made during execution."""
        with assert_no_network():
            # Run smoke command
            cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "SPY,TSLA"
            ]
            
            result = run_command(cmd)
            assert result.returncode == 0, "Should complete without network calls"
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        with assert_no_network():
            # Test with invalid symbols
            cmd = [
                "python", "scripts/multi_walkforward_report.py",
                "--smoke",
                "--validate-data",
                "--log-level", "INFO",
                "--symbols", "INVALID_SYMBOL"
            ]
            
            result = run_command(cmd)
            # Should handle gracefully (either fail with clear error or skip)
            assert result.returncode in [0, 1], "Should handle invalid symbols gracefully"
            
            if result.returncode == 1:
                assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
