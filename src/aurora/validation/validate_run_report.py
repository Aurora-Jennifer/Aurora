#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path("contracts/run_report.schema.json")


def main() -> None:
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/run.json")
    schema = json.loads(SCHEMA_PATH.read_text())
    report = json.loads(report_path.read_text())

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.path))
    if errors:
        for err in errors[:50]:
            loc = ".".join(map(str, err.path))
            print(f"[SCHEMA] {loc}: {err.message}")
        raise SystemExit(1)
    print(f"[SCHEMA] OK: {report_path}")


if __name__ == "__main__":
    main()


