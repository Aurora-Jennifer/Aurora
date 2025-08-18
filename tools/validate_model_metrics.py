import sys
import json
from pathlib import Path
from jsonschema import validate


def main():
    schema_path = Path("reports/model_metrics.schema.json")
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/model_drift.json")
    data = json.loads(input_path.read_text())
    schema = json.loads(schema_path.read_text())
    validate(instance=data, schema=schema)

    psi_global = float(data.get("psi_global", float("nan")))
    thr = data.get("thresholds", {})
    psi_fail = float(thr.get("psi_fail", 0.25))
    max_missing = float(thr.get("max_missing_pct", 0.01))

    # Missing rate checks
    for feat, st in data.get("stats", {}).items():
        miss = float(st.get("missing_pct", 0))
        if miss > max_missing:
            print(f"[MODEL EVAL FAIL] missing rate too high: {feat} = {miss:.4f} > {max_missing:.4f}")
            sys.exit(1)

    if not (psi_global <= psi_fail):
        print(f"[MODEL EVAL FAIL] psi_global {psi_global:.4f} > {psi_fail:.4f}")
        sys.exit(1)

    print("[MODEL EVAL OK]")


if __name__ == "__main__":
    main()


