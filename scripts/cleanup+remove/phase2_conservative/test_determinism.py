#!/usr/bin/env python3
"""
Determinism Gate: Ensure identical inputs produce identical outputs
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def hash_file(filepath: Path) -> str:
    """Generate SHA256 hash of file contents"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_determinism_test() -> bool:
    """Test that the E2E pipeline produces identical outputs with same seed"""

    # Set deterministic environment
    env = os.environ.copy()
    env.update({
        'PYTHONHASHSEED': '0',
        'TZ': 'UTC',
        'ML_SEED': '42',
        'ML_EXPORT_ONNX': '1'
    })

    results = []

    for run_id in range(2):
        print(f"[DETERMINISM] Run {run_id + 1}/2")

        # Run pipeline in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / f"run_{run_id}"
            run_dir.mkdir()

            # Copy config to run directory
            cmd = [
                sys.executable, 'scripts/train_linear.py',
                'golden_xgb_v2'
            ]

            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(run_dir),
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"[DETERMINISM][FAIL] Run {run_id + 1} failed")
                print(result.stderr)
                return False

            # Collect artifact hashes
            artifacts = {}

            # Check for key output files
            for pattern in ['artifacts/models/*.onnx', 'reports/experiments/*.json']:
                for filepath in run_dir.glob(pattern):
                    if filepath.is_file():
                        artifacts[filepath.name] = hash_file(filepath)

            results.append(artifacts)

    # Compare results
    if len(results) != 2:
        print("[DETERMINISM][FAIL] Did not get 2 complete runs")
        return False

    mismatches = []
    for filename in set(results[0].keys()) | set(results[1].keys()):
        hash1 = results[0].get(filename)
        hash2 = results[1].get(filename)

        if hash1 != hash2:
            mismatches.append(f"{filename}: {hash1} vs {hash2}")

    if mismatches:
        print("[DETERMINISM][FAIL] Output differences detected:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return False

    print("[DETERMINISM][PASS] Identical outputs across runs")
    return True


def main() -> int:
    """Main determinism test"""
    print("🎯 Testing determinism: same seed → same outputs")

    if not run_determinism_test():
        return 1

    print("✅ Determinism gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
