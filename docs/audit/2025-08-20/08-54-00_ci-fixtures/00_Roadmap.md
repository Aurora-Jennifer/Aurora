# CI Fixtures Fix — Roadmap (2025-08-20 08:54)
**Prompt:** "Fix CI failure due to missing smoke test fixtures"
## Context
- CI smoke test failing with 'No offline data available for SPY'
- Fixtures existed locally but weren't committed to repo
- .gitignore had exception but files weren't tracked
## Plan
- Force add smoke fixtures to git
- Commit and push to fix CI
## Success
- Smoke test passes locally with fixtures
- CI should now have required data files
