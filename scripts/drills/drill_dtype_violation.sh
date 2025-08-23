#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-artifacts/run}"
F="$(ls -1 "${OUT}"/*.parquet | head -n1)"

echo "[DRILL] Forcing a float64 column into ${F}"
python -c "
import sys, pandas as pd, numpy as np
p = sys.argv[1]
df = pd.read_parquet(p)
num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
assert num, 'no numeric columns'
c = num[0]
df[c] = df[c].astype('float64')
df.to_parquet(p)
print(f'[DRILL] Column {c} set to float64')
" "${F}"

echo "[EXPECT] dtype gate should FAIL"
if pytest -q tests/gates/l0/test_dtypes.py; then
  echo "❌ Drill failed: dtype test unexpectedly passed" >&2
  exit 1
else
  echo "✅ Drill passed: dtype test failed as expected"
fi
