#!/usr/bin/env bash
set -euo pipefail
if [[ "${BYPASS_SMOKE:-0}" == "1" ]]; then
  echo "[pre-push] Bypass enabled."
  exit 0
fi
echo "[pre-push] Running smoke..."
make smoke


