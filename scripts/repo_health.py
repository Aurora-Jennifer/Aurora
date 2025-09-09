#!/usr/bin/env python
"""
One-command health check gate - must pass before anything else
"""
import subprocess
import sys

CMDS = [
    ["ruff", "check", "."],
    ["mypy", "--strict", "src/"],
    ["pytest", "-q", "--maxfail=1", "--disable-warnings", "--cov=src", "--cov-report=term-missing:skip-covered"],
    ["pytest", "tests/training/", "-q", "--maxfail=1", "--disable-warnings"],
    ["bandit", "-q", "-r", "src"],
    ["deptry", "."],
    ["vulture", "src", "tests", ".vulture-whitelist.py", "--min-confidence", "80"],
]

def main():
    """Run all health checks"""
    for cmd in CMDS:
        print("$", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc:
            print(f"❌ Health check failed: {' '.join(cmd)}")
            sys.exit(rc)
    
    print("✅ HEALTHCHECK: OK")

if __name__ == "__main__":
    main()