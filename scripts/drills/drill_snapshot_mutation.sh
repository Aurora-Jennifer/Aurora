#!/usr/bin/env bash
set -euo pipefail

SNAP="${1:-artifacts/snapshots/golden_ml_v1}"
FILE="$(ls -1 "${SNAP}"/*.parquet | head -n1)"

echo "[DRILL] Mutating snapshot byte in: ${FILE}"
# flip one byte at end (will change HASH and RO might block)
chmod u+w "${FILE}" || true
python -c "
import sys, os
p = sys.argv[1]
with open(p, 'ab') as f:
    f.write(b'\x00')
print('[DRILL] Wrote one byte.')
" "${FILE}"

echo "[EXPECT] test_snapshot_hash_matches_manifest should FAIL"
if pytest -q tests/gates/l0/test_snapshot.py; then
  echo "❌ Drill failed: test unexpectedly passed" >&2
  exit 1
else
  echo "✅ Drill passed: test failed as expected"
fi
