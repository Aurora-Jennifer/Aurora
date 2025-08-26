#!/usr/bin/env bash
set -euo pipefail
echo "== L0 acceptance =="
python -u scripts/hash_snapshot.py artifacts/snapshots/golden_ml_v1 >/dev/null
make l0-gates
echo "✅ L0 acceptance passed"
