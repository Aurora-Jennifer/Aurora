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
    # Soft sanity checks
    if data.get("trades", 0) < 1:
        print("[PROMOTE FAIL] zero trades in smoke")
        return 1
    sharpe = float(data.get("sharpe", 0))
    if not (-10 <= sharpe <= 10):
        print("[PROMOTE FAIL] sharpe out of sane bounds")
        return 1
    print("[PROMOTE OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


