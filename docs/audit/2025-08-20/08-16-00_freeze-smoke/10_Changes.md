# Changes
## Actions
- config/data_sanity.yaml: add , set 
- scripts/walkforward_framework.py: emit SMOKE_OHLC_GUARD_OK cues
- Makefile: switch smoke to 
- .github/workflows/ci.yml: use smoke profile; run tiny smoke test; upload JUnit
- tests/smoke/test_smoke_datasanity_contract.py: new
## Commands
- make smoke
- pytest -q tests/smoke/test_smoke_datasanity_contract.py
