#!/usr/bin/env python3
import json
import os
from pathlib import Path

import pandas as pd

SNAP_DIR = Path("artifacts/snapshots/golden_ml_v1")


def main():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    symbols = ["SPY", "QQQ"]
    manifests = {}
    for sym in symbols:
        src = Path("data/raw") / f"{sym}.parquet"
        if not src.exists():
            raise SystemExit(f"Missing raw parquet: {src}")
        df = pd.read_parquet(src)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        # Optional DataSanity validation at ingest boundary (fail-fast behind flag)
        if os.getenv("DS_VALIDATE", "0") in ("1", "true", "TRUE", "yes"):
            try:
                from core.data_sanity import main as datasanity  # type: ignore
                datasanity.validate(df)
            except Exception as e:
                if os.getenv("DS_STRICT", "1") in ("1", "true", "TRUE", "yes"):
                    raise SystemExit(f"DataSanity validation failed for {sym}: {e}") from e
                print(f"[DS][WARN] validation error for {sym}: {e}; proceeding (non-strict)")
        # Freeze snapshot
        out = SNAP_DIR / f"{sym}.parquet"
        df.to_parquet(out)
        manifests[sym] = {"rows": int(len(df)), "cols": list(map(str, df.columns))}
    manifest = {"snapshot": "golden_ml_v1", "symbols": symbols, "tables": manifests}
    (SNAP_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"frozen: {SNAP_DIR}")


if __name__ == "__main__":
    main()


