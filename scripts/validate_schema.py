import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_columns(df: pd.DataFrame, spec: List[Dict[str, str]]) -> List[str]:
    errors: List[str] = []
    for col in spec:
        name = col["name"]
        dtype = col["dtype"]
        if name not in df.columns:
            errors.append(f"missing column: {name}")
            continue
        # dtype string match; allow pandas compatible equivalence
        actual = str(df[name].dtype)
        if dtype.startswith("datetime"):
            if not pd.api.types.is_datetime64_any_dtype(df[name]):
                errors.append(f"column {name} dtype {actual} != {dtype}")
        else:
            if dtype not in actual:
                errors.append(f"column {name} dtype {actual} != {dtype}")
    return errors


def main(argv: List[str]) -> int:
    # Minimal validator: checks schema file exists and is well-formed.
    root = Path(__file__).resolve().parents[1]
    schema_path = root / "config" / "data_schema.yaml"
    if not schema_path.exists():
        print("ERROR: config/data_schema.yaml missing", file=sys.stderr)
        return 2
    schema = load_yaml(schema_path)
    if "columns" not in schema or not isinstance(schema["columns"], list):
        print("ERROR: schema missing columns[]", file=sys.stderr)
        return 2
    if "missing_data" not in schema or not isinstance(schema["missing_data"], dict):
        print("ERROR: schema missing missing_data{}", file=sys.stderr)
        return 2

    # Optional: if a sample data snapshot exists, validate basic contract
    # Look for reports/smoke_run.json to extract a sample csv/parquet path if present
    run_meta = root / "reports" / "smoke_run.json"
    sample_validated = False
    if run_meta.exists():
        try:
            meta = json.loads(run_meta.read_text())
            sample_path = None
            for k in ("data_snapshot", "source_path"):
                if isinstance(meta.get(k), str):
                    sample_path = meta[k]
                    break
            if sample_path and Path(sample_path).exists():
                p = Path(sample_path)
                if p.suffix == ".csv":
                    df = pd.read_csv(p)
                elif p.suffix in (".parquet", ".pq"):
                    df = pd.read_parquet(p)
                else:
                    df = None
                if df is not None:
                    errs = validate_columns(df, schema["columns"])
                    if errs:
                        print("ERROR: schema validation errors:")
                        for e in errs:
                            print(f" - {e}")
                        return 1
                    sample_validated = True
        except Exception as ex:  # pragma: no cover
            print(f"WARN: sample validation skipped: {ex}")

    print(
        "Schema OK"
        + (" (sample validated)" if sample_validated else " (structure only)")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


