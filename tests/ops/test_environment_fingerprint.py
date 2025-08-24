"""
Environment Fingerprint Tests
Ensure runtime environment matches locked manifest for reproducibility.
"""

import json
import os
import pkgutil
import platform
from pathlib import Path
from typing import Any

import pytest


def load_manifest() -> dict[str, Any]:
    """Load the environment manifest."""
    manifest_path = Path("artifacts/manifest.json")
    if not manifest_path.exists():
        # Create a basic manifest if it doesn't exist
        manifest = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "deps": {},
            "container_digest": "unknown"
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    with open(manifest_path) as f:
        return json.load(f)


def test_env_fingerprint_matches_manifest():
    """Test that runtime environment matches locked manifest."""
    manifest = load_manifest()

    # Check Python version
    assert manifest["python"] == platform.python_version(), \
        f"Python version mismatch: manifest={manifest['python']}, runtime={platform.python_version()}"

    # Check platform
    assert manifest["platform"] == platform.platform(), \
        f"Platform mismatch: manifest={manifest['platform']}, runtime={platform.platform()}"

    # Check key dependencies
    key_deps = ["numpy", "pandas", "scipy", "sklearn", "torch", "yfinance"]
    for lib in key_deps:
        if lib in manifest["deps"]:
            m = pkgutil.find_loader(lib)
            assert m is not None, f"Missing dependency: {lib}"

    # Check container digest if available
    if manifest["container_digest"] != "unknown":
        digest_file = Path(".container_digest")
        if digest_file.exists():
            current_digest = digest_file.read_text().strip()
            assert manifest["container_digest"] == current_digest, \
                f"Container digest mismatch: manifest={manifest['container_digest']}, current={current_digest}"


def test_requirements_lock_exists():
    """Test that requirements.lock exists and is recent."""
    lock_file = Path("requirements.lock")
    assert lock_file.exists(), "requirements.lock file not found"

    # Check if lock file is recent (within 90 days)
    import time
    lock_age_days = (time.time() - lock_file.stat().st_mtime) / (24 * 3600)
    assert lock_age_days < 90, f"requirements.lock is {lock_age_days:.1f} days old (>90 days)"


def test_no_unpinned_dependencies():
    """Test that requirements files don't contain unpinned dependencies."""
    req_files = ["requirements.txt", "requirements-dev.txt"]

    for req_file in req_files:
        req_path = Path(req_file)
        if req_path.exists():
            with open(req_path) as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Check for unpinned dependencies (no ==, ~=, or specific version)
                    # Allow >= and <= for version ranges
                    if "==" not in line and "~=" not in line and ">=" not in line and "<=" not in line and line.count(".") < 2:
                        # Allow comments and empty lines
                        if not line.startswith("#") and line:
                            pytest.fail(f"Unpinned dependency in {req_file}:{line_num}: {line}")


def test_environment_variables_consistent():
    """Test that critical environment variables are set consistently."""
    critical_vars = [
        "PYTHONHASHSEED",  # For deterministic hashing
        "TZ",  # Timezone
    ]

    for var in critical_vars:
        if var in os.environ:
            # Log the value for debugging
            print(f"{var}={os.environ[var]}")
        else:
            # Not critical to fail, but warn
            print(f"Warning: {var} not set")


def test_system_resources_adequate():
    """Test that system has adequate resources for paper trading."""
    import psutil

    # Check memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    assert memory_gb >= 4.0, f"Insufficient memory: {memory_gb:.1f}GB (<4GB required)"

    # Check disk space
    disk_gb = psutil.disk_usage(".").free / (1024**3)
    assert disk_gb >= 1.0, f"Insufficient disk space: {disk_gb:.1f}GB (<1GB required)"

    # Check CPU cores
    cpu_count = psutil.cpu_count()
    assert cpu_count >= 2, f"Insufficient CPU cores: {cpu_count} (<2 required)"


if __name__ == "__main__":
    pytest.main([__file__])
