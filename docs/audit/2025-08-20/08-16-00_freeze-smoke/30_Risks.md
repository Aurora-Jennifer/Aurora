# Risks & Rollback
- Risk: cue strings get refactored → brittle test
- Mitigation: keep single stable cues; update test if needed
- Rollback: git checkout -p config/data_sanity.yaml scripts/walkforward_framework.py Makefile .github/workflows/ci.yml tests/smoke/test_smoke_datasanity_contract.py
