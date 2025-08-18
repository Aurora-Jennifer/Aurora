---
name: Bug report
about: Report a reproducible issue
labels: bug
---

**What happened**
<describe the failure>

**Repro**
- commit: `<git sha>` 
- command: `make smoke` | `pytest -q ...`
- artifacts: attach `reports/smoke_run.json` or failing test output

**Expected**
<what you expected>

**Environment**
- Python: `python --version`
- OS: <…>

**Notes**
- violation_code: `<if present>`
- logs: <fold-level lines only, no secrets>

