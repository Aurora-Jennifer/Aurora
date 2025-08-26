#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def get_git_info() -> dict[str, str]:
    """Get git commit and branch info."""
    try:
        commit = subprocess.getoutput("git rev-parse --short HEAD") or "unknown"
        branch = subprocess.getoutput("git branch --show-current") or "unknown"
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}

def get_build_info() -> dict[str, str]:
    """Get build environment info."""
    import sys
    try:
        import onnx
        import xgboost as xgb
        xgb_ver = xgb.__version__
        onnx_ver = onnx.__version__
    except ImportError:
        xgb_ver = "unknown"
        onnx_ver = "unknown"

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "xgboost": xgb_ver,
        "onnx": onnx_ver,
        "onnx_opset": "13"  # from export
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/gate_promote.py <model_version_dir>")
        print("Example: python scripts/gate_promote.py artifacts/models/1755790459")
        sys.exit(1)

    ver_dir = Path(sys.argv[1])
    if not ver_dir.exists():
        print(f"[PROMOTE] ❌ Model directory not found: {ver_dir}")
        sys.exit(1)

    print(f"[PROMOTE] Promoting {ver_dir.name}...")

    # 1) Verify parity
    parity_path = ver_dir / "parity.json"
    if not parity_path.exists():
        print(f"[PROMOTE] ❌ Missing parity.json in {ver_dir}")
        sys.exit(1)

    with open(parity_path) as f:
        parity = json.load(f)
    if not parity.get("ok", False):
        print("[PROMOTE] ❌ Refusing to promote: parity not ok")
        print(f"  max_abs_diff: {parity.get('max_abs_diff', 'unknown')}")
        print(f"  nrows: {parity.get('nrows', 'unknown')}")
        sys.exit(1)

    # 2) Verify required files exist
    required_files = ["model.onnx", "parity.json", "bench.json", "sidecar.json"]
    missing_files = []
    for f in required_files:
        if not (ver_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"[PROMOTE] ❌ Missing required files: {missing_files}")
        sys.exit(1)

    # 3) Generate manifest with hashes
    manifest = {
        "exp_id": ver_dir.name,
        "snapshot": "golden_ml_v1",  # TODO: extract from sidecar
        "profile": "golden_xgb_v2",  # TODO: extract from sidecar
        "created_at": subprocess.getoutput("date -u +%Y-%m-%dT%H:%M:%SZ") or "unknown",
        "git": get_git_info(),
        "build": get_build_info(),
        "files": {}
    }

    # Compute SHA256 for all files
    for f in required_files:
        p = ver_dir / f
        manifest["files"][f] = {
            "sha256": sha256(p),
            "bytes": p.stat().st_size
        }

    # Write manifest
    manifest_path = ver_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[PROMOTE] ✅ Manifest written: {manifest_path}")

    # 4) Atomic symlink flip
    latest = ver_dir.parent / "latest"
    tmp = ver_dir.parent / ".latest.tmp"

    # Clean up any existing temp symlink
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()

    # Create temp symlink
    try:
        os.symlink(ver_dir.name, tmp)
    except Exception as e:
        print(f"[PROMOTE] ❌ Failed to create temp symlink: {e}")
        sys.exit(1)

    # Atomic replace
    try:
        os.replace(tmp, latest)
        print(f"[PROMOTE] ✅ Atomic flip: {ver_dir.name} -> latest")
    except Exception as e:
        print(f"[PROMOTE] ❌ Failed atomic flip: {e}")
        if tmp.exists():
            tmp.unlink()
        sys.exit(1)

    # 5) Log promotion
    log_path = Path("artifacts/promotions.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        timestamp = subprocess.getoutput("date -u +%Y-%m-%dT%H:%M:%SZ") or "unknown"
        f.write(f"{timestamp} promote {ver_dir.name} ok sha256={manifest['files']['model.onnx']['sha256'][:8]}...\n")

    print(f"[PROMOTE] ✅ Promotion logged to {log_path}")
    print(f"[PROMOTE] 🎉 Successfully promoted {ver_dir.name}")

if __name__ == "__main__":
    main()


