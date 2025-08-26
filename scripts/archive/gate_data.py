import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fail(msg: str) -> int:
    print(f"[GATE:DATA][FAIL] {msg}")
    return 1


def ok(msg: str) -> int:
    print(f"[GATE:DATA][OK] {msg}")
    return 0


def check_schema_dtype(features_path: Path, sidecar_path: Path) -> None:
    if not features_path.exists() or not sidecar_path.exists():
        raise SystemExit(fail(f"Missing inputs: features={features_path.exists()} sidecar={sidecar_path.exists()}"))
    df = pd.read_parquet(features_path)
    side = json.loads(sidecar_path.read_text())
    expected_cols = side.get("features") or side.get("feature_names")
    if expected_cols is None:
        raise SystemExit(fail("sidecar missing 'features'"))
    cols = list(df.columns)
    if cols != expected_cols:
        raise SystemExit(fail(f"schema mismatch. expected={expected_cols} actual={cols}"))
    # dtype check: float32
    bad = [c for c in df.columns if str(df[c].dtype) != "float32"]
    if bad:
        raise SystemExit(fail(f"non-float32 columns: {bad}"))
    # NaN check
    if int(df.isna().sum().sum()) > 0:
        raise SystemExit(fail("NaNs present in features"))
    print("SMOKE_FEATURE_SCHEMA_OK")


def check_determinism(features_path: Path) -> None:
    # Re-read twice and compare hashes
    df1 = pd.read_parquet(features_path)
    df2 = pd.read_parquet(features_path)
    x1 = np.ascontiguousarray(df1.values).view("u1").data.tobytes()
    x2 = np.ascontiguousarray(df2.values).view("u1").data.tobytes()
    if x1 != x2:
        raise SystemExit(fail("feature matrix not deterministic across reads"))
    print("SMOKE_DETERMINISM_OK")


def check_chronology(meta_path: Path) -> None:
    # Soft structural check: ensure purged scheme + gap present
    if not meta_path.exists():
        raise SystemExit(fail("missing search_history.json for chronology check"))
    h = json.loads(meta_path.read_text())
    early = h.get("meta", {}).get("early_stopping", {})
    if early.get("policy") != "train_tail":
        raise SystemExit(fail("early_stopping policy must be 'train_tail'"))
    cv = h.get("meta", {}).get("cv", {})
    if not (cv.get("scheme") == "purged_kfold" and int(cv.get("purge_gap_bars", 0)) >= 1):
        raise SystemExit(fail("CV must be purged_kfold with purge_gap_bars >= 1"))
    print("SMOKE_CHRONOLOGY_OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/parity/features_oof.parquet")
    ap.add_argument("--sidecar", default="artifacts/parity/sidecar.json")
    ap.add_argument("--history", default="reports/experiments/search_history.json")
    args = ap.parse_args()

    try:
        check_schema_dtype(Path(args.features), Path(args.sidecar))
        check_determinism(Path(args.features))
        check_chronology(Path(args.history))
    except SystemExit as e:
        # surface non-zero codes
        return e.code if isinstance(e.code, int) else 1
    return ok("data gates passed")


if __name__ == "__main__":
    raise SystemExit(main())


