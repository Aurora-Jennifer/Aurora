#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

try:
    from jsonschema import ValidationError, validate
except Exception:
    print("jsonschema not installed. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--schema", required=True)
    a = p.parse_args()

    data = json.loads(pathlib.Path(a.input).read_text())
    schema = json.loads(pathlib.Path(a.schema).read_text())
    try:
        validate(data, schema)
    except ValidationError as e:
        print(f"[PROMOTE FAIL] {e.message}")
        return 1

    status = data.get("status")
    if status == "FAIL":
        code = data.get("violation_code", "UNKNOWN")
        print(f"[PROMOTE FAIL] status=FAIL code={code}")
        return 1

    # status OK path
    fold_summaries = data.get("fold_summaries", [])
    if not fold_summaries:
        print("[PROMOTE FAIL] missing fold_summaries on OK status")
        return 1

    # Soft sanity checks
    if data.get("trades", 0) < 1:
        print("[PROMOTE FAIL] zero trades in smoke")
        return 1
    sharpe = float(data.get("sharpe", 0))
    if not (-10 <= sharpe <= 10):
        print("[PROMOTE FAIL] sharpe out of sane bounds")
        return 1
    # Phase timing soft SLA
    MAX_TOTAL_MS = 60_000
    pt = data.get("phase_times_ms", {})
    total_ms = sum(pt.values()) if pt else float(data.get("duration_s", 0)) * 1000.0
    if total_ms > MAX_TOTAL_MS:
        print(f"[PROMOTE FAIL] runtime budget exceeded: {total_ms:.0f}ms > {MAX_TOTAL_MS}ms")
        return 1
    # Model provenance (if present, must be complete)
    if "model" in data:
        for k in ("id", "kind", "artifact_sha256"):
            if not data["model"].get(k):
                print(f"[PROMOTE FAIL] missing model.{k}")
                return 1
    print("[PROMOTE OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
