# Next Session Plan — 2025-08-20

## Goal
Promotion gate, artifact alignment, and minimal docs/tests to lock in the new rules with small diffs.

## Plan (minimal diffs)
1) CI/promotion gate
   - Add a promote job in .github/workflows/ci.yml (needs: smoke).
   - New scripts/validate_run_hashes.py to assert:
     - reports/run.json.config_hash == sha256(active config bundle)
     - reports/run.json.data_hash == expected snapshot id
   - Promotion fails on mismatch; run Bandit here as non-allow-fail.
2) Deterministic deps
   - Switch CI installs to use requirements.lock (or pin requirements.txt to ==).
   - Add tools/lock_refresh.sh; add audit TODO if lock >90d.
3) Artifacts alignment
   - Make smoke also write reports/run.json (alias of current smoke_run.json) with: run_id, config_hash, data_hash, started_at, finished_at.
4) Docs/runbooks
   - Add minimal README to missing top-level folders with: Purpose:, Entrypoints:, Do-not-touch:.
   - Add docs/summaries/engine.md, docs/summaries/data_sanity.md with a Mermaid diagram each.
5) Boundaries follow-up
   - Decide: keep dynamic imports in scripts/ or migrate entrypoints to cli/ and leave scripts/ for shell.
   - If moving, update Makefile/docs accordingly.
6) Tests (quick wins)
   - Unit: rate-limit wrapper (caps at N/min and raises clearly).
   - Integration: kill.flag causes scripts/canary_runner.py to exit within one tick.
7) CONTRIBUTING ergonomics
   - Document pre-push smoke hook install and BYPASS_SMOKE=1 escape hatch.
8) Security gating in release
   - In release.yml, ensure promotion path fails on Bandit MEDIUM+ (routine security.yml can remain allow-pass).

## Order of execution (1–2 hours)
1) Promotion job + run-hash validator (30–40m)
2) Switch CI to requirements.lock (15m)
3) Smoke writes reports/run.json alias (15m)
4) Tests (20–30m)
5) One or two folder READMEs + one summary doc (15–20m)
6) Decide scripts->cli migration and log TODO (remaining time)

## Quick commands
- make smoke
- python scripts/validate_schema.py
- pytest -q -k "rate_limit or canary_runner" | cat
- git add -p && git commit -m "ci: promotion gate + artifacts alignment (audit 2025-08-20)"

## Notes
- Keep diffs surgical; avoid protected paths.
- Treat promotion as the only blocking gate; lint/tests-full remain allow-fail.
