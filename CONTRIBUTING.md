# Contributing to Aurora

## Quickstart
- Python 3.11
- `pip install -r requirements.txt`
- (optional) `pip install pre-commit && pre-commit install`

## Everyday Commands
- Smoke (fast e2e): `make smoke`
- Data sanity tests only: `make datasanity`
- Golden regression: `make golden`
- (Re)bless golden baseline (intentional change): `make bless_golden`
- Quality (coverage report): `make quality`
- Config schema check: `make configcheck`

## Smoke Contract (CI)
- CI Smoke runs with `--datasanity-profile walkforward_smoke --allow-zero-trades`.
- The smoke path enforces finite float64 OHLC and logs a stable cue:
  - `SMOKE_OHLC_GUARD_OK: train fold finite float64 enforced`
  - `SMOKE_OHLC_GUARD_OK: test fold finite float64 enforced`
- A dedicated test `tests/smoke/test_smoke_datasanity_contract.py` asserts these cues and the profile.
- Do not change Smoke flags or remove the cues without updating the test.

## CI Facts
- Required checks on `main`: **Smoke / smoke**, **Smoke / Promote gate**
- Non-blocking signals: datasanity, golden, quality, security (Bandit, Gitleaks, pip-audit)
- Tag releases run Smoke first (see `.github/workflows/release.yml`)

## Rules of the Road
- Follow the MASTER RULESET + companion rules in Cursor.
- No secrets in code, logs, or audit. Redact as `[redacted]`.
- Determinism: use the offline cache in CI; don’t depend on live data.
- If DataSanity fails, return structured `{status:"FAIL", violation_code, reason}` (no tracebacks).
- CI enforces risk guards: costs, leverage, gross & per-position exposure.

## PR Checklist
- [ ] Smoke is green locally (`make smoke`)
- [ ] DataSanity impact understood (violation codes if any)
- [ ] Tests added/updated (datasanity/golden where relevant)
- [ ] Docs updated (runbook or changelog if behavior changed)

## Releasing
1. Ensure `main` is green.
2. Tag semver: `git tag vX.Y.Z && git push --tags`
3. GitHub Actions will run Smoke and then draft the release if green.

## Troubleshooting
- **Smoke FAIL**: open `reports/smoke_run.json` for `violation_code` + `reason`
- **Budget exceeded**: reduce symbols or windows; check `phase_times_ms` in JSON
- **Golden drift**: if intentional, run `make bless_golden` and commit the new baseline
