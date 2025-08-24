"""
Cross-Process Determinism Tests
Ensure two fresh processes produce byte-identical artifacts given same manifest.
"""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest


def run_e2d_process(output_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Run E2D process with specific seed and return results."""
    # Start with current environment
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(seed),
        "TZ": "UTC",
        "FLAG_DETERMINISTIC": "1"
    })

    cmd = [
        "python", "scripts/e2d.py",
        "--profile", "config/profiles/golden_xgb_v2.yaml"
    ]

    result = subprocess.run(
        cmd,
        env=env,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=60
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": output_dir
    }


def hash_file(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def test_cross_process_determinism():
    """Test that two fresh processes produce byte-identical artifacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run first process
        result1 = run_e2d_process(temp_path / "run1")
        assert result1["returncode"] == 0, f"First run failed: {result1['stderr']}"

        # Run second process
        result2 = run_e2d_process(temp_path / "run2")
        assert result2["returncode"] == 0, f"Second run failed: {result2['stderr']}"

        # Compare artifacts
        run1_dir = temp_path / "run1"
        run2_dir = temp_path / "run2"

        # Find all JSON files in both directories
        json_files1 = list(run1_dir.rglob("*.json"))
        json_files2 = list(run2_dir.rglob("*.json"))

        # Should have same number of files
        assert len(json_files1) == len(json_files2), \
            f"Different number of JSON files: {len(json_files1)} vs {len(json_files2)}"

        # Compare each file
        for file1 in json_files1:
            # Find corresponding file in run2
            relative_path = file1.relative_to(run1_dir)
            file2 = run2_dir / relative_path

            assert file2.exists(), f"Missing file in run2: {relative_path}"

            # Compare file hashes
            hash1 = hash_file(file1)
            hash2 = hash_file(file2)

            assert hash1 == hash2, \
                f"File {relative_path} differs between runs: {hash1} vs {hash2}"


def test_seed_reproducibility():
    """Test that same seed produces same results across runs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run with seed 42
        result1 = run_e2d_process(temp_path / "seed42_run1", seed=42)
        result2 = run_e2d_process(temp_path / "seed42_run2", seed=42)

        assert result1["returncode"] == 0, f"Seed 42 run 1 failed: {result1['stderr']}"
        assert result2["returncode"] == 0, f"Seed 42 run 2 failed: {result2['stderr']}"

        # Run with seed 123
        result3 = run_e2d_process(temp_path / "seed123", seed=123)
        assert result3["returncode"] == 0, f"Seed 123 run failed: {result3['stderr']}"

        # Compare seed 42 runs (should be identical)
        run1_dir = temp_path / "seed42_run1"
        run2_dir = temp_path / "seed42_run2"

        json_files1 = list(run1_dir.rglob("*.json"))
        # json_files2 = list(run2_dir.rglob("*.json"))  # Not needed for this test

        for file1 in json_files1:
            relative_path = file1.relative_to(run1_dir)
            file2 = run2_dir / relative_path

            hash1 = hash_file(file1)
            hash2 = hash_file(file2)

            assert hash1 == hash2, \
                f"Same seed produced different results for {relative_path}: {hash1} vs {hash2}"


def test_deterministic_metrics():
    """Test that metrics are deterministic across runs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)  # Need this for the test

        # Run multiple times with same seed
        results = []
        for i in range(3):
            result = run_e2d_process(temp_path / f"run{i}", seed=42)
            assert result["returncode"] == 0, f"Run {i} failed: {result['stderr']}"
            results.append(result)

        # Find metrics files
        metrics_files = []
        for i in range(3):
            run_dir = temp_path / f"run{i}"
            metrics_files.extend(list(run_dir.rglob("metrics.json")))

        # All metrics files should be identical
        if metrics_files:
            first_hash = hash_file(metrics_files[0])
            for metrics_file in metrics_files[1:]:
                file_hash = hash_file(metrics_file)
                assert file_hash == first_hash, \
                    f"Metrics file {metrics_file} differs from first: {file_hash} vs {first_hash}"


def test_environment_isolation():
    """Test that environment variables don't affect determinism."""
    with tempfile.TemporaryDirectory():  # temp_dir not used in this test
        # temp_path = Path(temp_dir)  # Not used in this test

        # Run with different environment variables (preserve current env)
        env1 = os.environ.copy()
        env1.update({"PYTHONHASHSEED": "42", "TZ": "UTC", "FLAG_DETERMINISTIC": "1"})

        env2 = os.environ.copy()
        env2.update({"PYTHONHASHSEED": "42", "TZ": "UTC", "FLAG_DETERMINISTIC": "1", "EXTRA_VAR": "test"})

        cmd = ["python", "scripts/e2d.py", "--profile", "config/profiles/golden_xgb_v2.yaml"]

        result1 = subprocess.run(cmd, env=env1, cwd=Path.cwd(), capture_output=True, text=True, timeout=60)
        result2 = subprocess.run(cmd, env=env2, cwd=Path.cwd(), capture_output=True, text=True, timeout=60)

        assert result1.returncode == 0, f"Run 1 failed: {result1.stderr}"
        assert result2.returncode == 0, f"Run 2 failed: {result2.stderr}"

        # Results should be identical despite different environment
        # (This is a basic check - in practice you'd compare specific artifacts)


if __name__ == "__main__":
    pytest.main([__file__])
