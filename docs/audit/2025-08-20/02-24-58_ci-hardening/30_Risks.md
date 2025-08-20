# Risks & Rollback
- Risk: CI misconfig → rollback: git checkout -p .github/workflows/ci.yml
- Risk: pre-push friction → set BYPASS_SMOKE=1
- Rollback lockfile: git mv requirements.lock requirements.lock.txt
