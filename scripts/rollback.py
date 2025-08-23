#!/usr/bin/env python3
"""
One-command rollback system for model registry.
Usage: python scripts/rollback.py <model_version>
Example: python scripts/rollback.py 1755790458
"""
import argparse
import builtins
import contextlib
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rollback to previous model version")
    parser.add_argument("version", help="Model version to rollback to (e.g., 1755790458)")
    parser.add_argument("--reason", default="manual_rollback", help="Reason for rollback")
    args = parser.parse_args()

    version = args.version
    models_dir = Path("artifacts/models")
    latest_link = models_dir / "latest"
    target_dir = models_dir / version

    # Validate target exists
    if not target_dir.exists():
        print(f"[ROLLBACK] ❌ Target version {version} not found in {models_dir}")
        print("Available versions:")
        for d in sorted(models_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir() and d.name != "latest":
                print(f"  {d.name}")
        sys.exit(1)

    # Validate target has required files
    required_files = ["model.onnx", "manifest.json", "parity.json"]
    missing = [f for f in required_files if not (target_dir / f).exists()]
    if missing:
        print(f"[ROLLBACK] ❌ Target missing required files: {missing}")
        sys.exit(1)

    # Get current version
    current_version = None
    if latest_link.exists() and latest_link.is_symlink():
        with contextlib.suppress(builtins.BaseException):
            current_version = latest_link.resolve().name

    print(f"[ROLLBACK] Rolling back from {current_version or 'unknown'} to {version}")

    # Atomic symlink flip
    tmp_link = models_dir / ".latest.tmp"

    # Clean up any existing temp symlink
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()

    try:
        # Create temp symlink
        os.symlink(version, tmp_link)

        # Atomic replace
        os.replace(tmp_link, latest_link)
        print(f"[ROLLBACK] ✅ Atomic flip: latest -> {version}")

    except Exception as e:
        print(f"[ROLLBACK] ❌ Failed atomic flip: {e}")
        if tmp_link.exists():
            tmp_link.unlink()
        sys.exit(1)

    # Log rollback
    log_path = Path("artifacts/promotions.log")
    if log_path.exists():
        with open(log_path, "a") as f:
            timestamp = subprocess.getoutput("date -u +%Y-%m-%dT%H:%M:%SZ") or "unknown"
            f.write(f"{timestamp} rollback {version} reason={args.reason}\n")
        print(f"[ROLLBACK] ✅ Rollback logged to {log_path}")

    # Verify rollback
    if latest_link.exists() and latest_link.is_symlink():
        try:
            actual_version = latest_link.resolve().name
            if actual_version == version:
                print(f"[ROLLBACK] 🎉 Successfully rolled back to {version}")
            else:
                print(f"[ROLLBACK] ⚠️  Warning: latest points to {actual_version}, expected {version}")
        except Exception as e:
            print(f"[ROLLBACK] ⚠️  Warning: could not verify rollback: {e}")
    else:
        print("[ROLLBACK] ❌ Rollback failed: latest symlink not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
