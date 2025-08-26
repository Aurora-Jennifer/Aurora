import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_active_config_hash() -> str:
    # Minimal: hash base.yaml; can be extended to include overlays/bundle
    base = Path("config/base.yaml")
    if not base.exists():
        print("ERROR: config/base.yaml missing", file=sys.stderr)
        return ""
    return sha256_file(base)


def main() -> int:
    run_path = Path("reports/run.json")
    if not run_path.exists():
        print("ERROR: reports/run.json not found", file=sys.stderr)
        return 2
    try:
        run = json.loads(run_path.read_text())
    except Exception as e:
        print(f"ERROR: failed to read run.json: {e}", file=sys.stderr)
        return 2

    required = [
        "run_id",
        "config_hash",
        "data_hash",
        "started_at",
        "finished_at",
    ]
    missing = [k for k in required if k not in run]
    if missing:
        print(f"ERROR: run.json missing keys: {', '.join(missing)}", file=sys.stderr)
        return 2

    want_cfg = compute_active_config_hash()
    if not want_cfg:
        return 2
    have_cfg = str(run.get("config_hash", ""))
    if have_cfg != want_cfg:
        print(
            f"ERROR: config_hash mismatch: run={have_cfg[:12]} active={want_cfg[:12]}",
            file=sys.stderr,
        )
        return 1

    # Optional: enforce data hash if EXPECT_DATA_HASH is set
    expect_data = os.getenv("EXPECT_DATA_HASH", "").strip()
    have_data = str(run.get("data_hash", ""))
    if expect_data:
        if have_data != expect_data:
            print(
                f"ERROR: data_hash mismatch: run={have_data[:12]} expected={expect_data[:12]}",
                file=sys.stderr,
            )
            return 1
    else:
        if not have_data:
            print("WARN: data_hash empty; set EXPECT_DATA_HASH to enforce")

    print("run.json hash validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())


