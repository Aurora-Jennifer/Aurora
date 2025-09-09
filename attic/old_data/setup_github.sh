#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-$(basename "$(git rev-parse --show-toplevel)")}" 
ORG_FLAG="${2:-}"
DEFAULT_BRANCH="main"

echo "Repo: $REPO_NAME"
echo "Creating PRIVATE repo on GitHub…"
gh repo create "$REPO_NAME" $ORG_FLAG --private --source=. --remote=origin --push || {
  echo "If repo exists, just setting remote…"
  if ! git remote | grep -q '^origin$'; then
    gh repo view "$REPO_NAME" $ORG_FLAG --json url -q .url >/dev/null
    URL=$(gh repo view "$REPO_NAME" $ORG_FLAG --json url -q .url)
    git remote add origin "$URL"
  fi
  git push -u origin HEAD || true
}

# Ensure default branch name
git branch -M "$DEFAULT_BRANCH"
git push -u origin "$DEFAULT_BRANCH" || true

echo "Setting branch protection on '$DEFAULT_BRANCH'…"
gh api \
  -X PUT \
  "repos/{owner}/$REPO_NAME/branches/$DEFAULT_BRANCH/protection" \
  -F required_status_checks.strict=true \
  -F required_status_checks.contexts[]="Tests" \
  -F required_status_checks.contexts[]="Promotion gate" \
  -F enforce_admins=true \
  -F required_pull_request_reviews.required_approving_review_count=0 \
  -F restrictions= | sed -n '1,80p' >/dev/null || true

echo "Triggering CI with a tiny commit if needed…"
if ! git diff --quiet; then
  git add -A
  git commit -m "chore: bootstrap CI"
fi
git push origin "$DEFAULT_BRANCH"

echo "Done.
Next:
1) Open GitHub → Actions tab → watch 'CI' job run.
2) Confirm blocking steps pass: 'Lint (changed files)', 'Tests', 'Promotion gate'.
3) Confirm artifact 'ruff-full' is attached.
"


