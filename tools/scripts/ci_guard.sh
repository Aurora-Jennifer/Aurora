#!/usr/bin/env bash
set -euo pipefail

echo "==> CI Guard: lint-changed, tests, promotion gate, and backup"

# Ensure we're in a Git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo"; exit 1; }

# Fetch base to compute changed files
BASE_REMOTE="origin"
BASE_BRANCH="main"
git fetch "$BASE_REMOTE" "$BASE_BRANCH" >/dev/null 2>&1 || true

# Lint only changed Python files vs origin/main
CHANGED=$(git diff --name-only --diff-filter=ACMRT "${BASE_REMOTE}/${BASE_BRANCH}...HEAD" | grep -E '\.py$' || true)
if [[ -n "$CHANGED" ]]; then
  echo "==> Ruff (changed files)"
  ruff check --output-format=github $CHANGED
else
  echo "==> No changed Python files for lint"
fi

echo "==> Pytest"
pytest -q

echo "==> Promotion gate"
make promote

# Backup (git bundle of current HEAD)
echo "==> Creating local backup bundle"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p artifacts/backups
git bundle create "artifacts/backups/${TS}_HEAD.bundle" HEAD
git rev-parse --short HEAD > artifacts/backups/latest.txt
echo "Backup written to artifacts/backups/${TS}_HEAD.bundle"

echo "==> CI Guard passed"
exit 0


