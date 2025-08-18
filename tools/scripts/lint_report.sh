#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts
ruff check . | tee artifacts/ruff_full.txt
echo "Wrote artifacts/ruff_full.txt"


