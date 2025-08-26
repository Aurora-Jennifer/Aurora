#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

from jsonschema import Draft7Validator


def load_json(path: pathlib.Path):
    with path.open() as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Path to reports/run.json")
    p.add_argument("--folds", required=True, help="Path to reports/folds directory")
    p.add_argument("--schema", default="reports/metrics.schema.json", help="Metrics schema path")
    args = p.parse_args()

    run_path = pathlib.Path(args.run)
    folds_dir = pathlib.Path(args.folds)
    schema_path = pathlib.Path(args.schema)

    schema = load_json(schema_path)
    validator = Draft7Validator(schema)

    errors = []

    # Validate run.json
    try:
        run_json = load_json(run_path)
        for e in sorted(validator.iter_errors(run_json), key=lambda e: e.path):
            errors.append(f"run.json: {list(e.path)}: {e.message}")
    except Exception as ex:
        errors.append(f"run.json: failed to load/parse: {ex}")

    # Validate folds
    if not folds_dir.exists():
        errors.append(f"folds dir missing: {folds_dir}")
    else:
        for fp in sorted(folds_dir.glob("*.json")):
            try:
                j = load_json(fp)
                for e in sorted(validator.iter_errors(j), key=lambda e: e.path):
                    errors.append(f"{fp.name}: {list(e.path)}: {e.message}")
            except Exception as ex:
                errors.append(f"{fp.name}: failed to load/parse: {ex}")

    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("Metrics validation passed.")

if __name__ == "__main__":
    main()
