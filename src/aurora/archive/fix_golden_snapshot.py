#!/usr/bin/env python
"""
Fix golden snapshot to meet L0 contracts:
- tz-aware UTC datetime index
- float32 numeric dtypes
- monotonic index; deduped
- optional symbol handling
Usage:
  python -u scripts/fix_golden_snapshot.py artifacts/snapshots/golden_ml_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SNAP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/snapshots/golden_ml_v1")
FEATURE_FILES = ("features.parquet", "X.parquet")  # will fix first one found

def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    # Promote a timestamp column to index if present
    for col in ("timestamp","ts","datetime","date"):
        if col in df.columns:
            df = df.set_index(col)
            break
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            raise SystemExit(f"Cannot coerce index to datetime: {e}")
    # Force tz-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    # Deduplicate & sort (stable)
    before = len(df)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    dropped = before - len(df)
    if dropped:
        print(f"[fix] dropped {dropped} duplicate index rows")
    # Enforce float32 numeric policy
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float32")
    # Finite check (fail loud if bad)
    arr = df.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise SystemExit(f"Non-finite values detected at positions {list(zip(*bad, strict=False))[:5]}")
    return df

def main():
    SNAP.mkdir(parents=True, exist_ok=True)
    # choose first existing feature file
    src = None
    for name in FEATURE_FILES:
        p = SNAP / name
        if p.exists():
            src = p
            break
    if src is None:
        raise SystemExit(f"No feature file found in {SNAP} among {FEATURE_FILES}")
    df = load_df(src)
    # Write back atomically
    tmp = SNAP / (src.stem + ".tmp.parquet")
    df.to_parquet(tmp, index=True)
    tmp.replace(src)
    print(f"[ok] fixed {src} → tz UTC, float32, monotonic, deduped")

if __name__ == "__main__":
    main()
