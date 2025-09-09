#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-artifacts/run}"
F="$(ls -1 "${OUT}"/*.parquet | head -n1)"

echo "[DRILL] Removing tz (UTC) from index in ${F}"
python -c "
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_parquet(p)
for col in ('timestamp','ts','datetime','date'):
    if col in df.columns:
        df = df.set_index(col)
        break
idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
df.index = idx  # naive timestamps
df.to_parquet(p)
print('[DRILL] Index made naive (no tz).')
" "${F}"

echo "[EXPECT] timezone gate should FAIL"
if pytest -q tests/gates/l0/test_timezones.py::test_index_is_tz_aware_utc; then
  echo "❌ Drill failed: timezone test unexpectedly passed" >&2
  exit 1
else
  echo "✅ Drill passed: timezone test failed as expected"
fi
