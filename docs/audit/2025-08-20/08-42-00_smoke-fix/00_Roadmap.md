# Smoke Fix — Roadmap (2025-08-20 08:42)
**Prompt:** "Fix smoke test hanging issue"
## Context
- Smoke test was hanging without --allow-zero-trades flag
- DataSanity validation working correctly (0 repairs, 0 outliers)
## Plan
- Identify missing flag support
- Test with correct arguments
- Push fix to CI
## Success
- Smoke test completes with exit code 0
- CI and Makefile already had correct flags
- All DataSanity tests pass locally
