# ✅ Aurora Compliance Checklist

> Snapshot: update when rulesets change. Goal: quick, human-readable proof that we’re operating under the charter.

## Core Gates
- [ ] Smoke CI green on PRs & main (required)
- [ ] Promote gate validates `reports/smoke_run.json` (status, fold_summaries, bounds)
- [ ] Branch protection requires “Smoke / smoke” and “Smoke / Promote gate”

## DataSanity (walkforward/backtest)
- [ ] Enforce profile `walkforward_ci` in CI
- [ ] Violation codes surfaced: DUP_TS, NON_MONO_INDEX, NAN_VALUES, INF_VALUES, EMPTY_SERIES, SHORT_FOLD
- [ ] Deterministic unit tests for each code (no network)
- [ ] Docs: Failure Codes table in `docs/runbooks/smoke.md`

## Golden Regression
- [ ] Deterministic `tests/golden/SPY.parquet` generated in CI
- [ ] Baseline `baselines/spy_golden.json` blessed and guarded (ε bounds)
- [ ] `make golden` job present in CI (non-blocking/blocked? ____)

## Governance & Security
- [ ] Secrets/redaction policy in place (no secrets in logs/audit)
- [ ] Bandit & Gitleaks run on PRs (non-blocking/blocked? ____)
- [ ] Dependencies pinned & `requirements.lock` updated regularly

## Observability
- [ ] `reports/smoke_run.json` (contract) + `reports/smoke_run.meta.json` (git_sha, profile, symbols, folds)
- [ ] One line per fold (no per-bar spam)
- [ ] ntfy/Slack notifications wired (success/fail/kill-switch)

## Ergonomics
- [ ] `make smoke`, `make datasanity`, `make golden`, `make bless_golden`
- [ ] Pre-push guard runs smoke locally
- [ ] PR template in `.github/PULL_REQUEST_TEMPLATE.md`

## Release
- [ ] `release.yml` requires smoke on tag
- [ ] Changelog updated on release
- [ ] Rollback steps documented in `docs/runbooks/smoke.md`

## Documentation
- [ ] README has real smoke badge (`<owner>/<repo>`)
- [ ] Runbooks: smoke + golden + recovery
- [ ] MODULE_MAP.md (entrypoints, public APIs) up to date

**Last reviewed:** YYYY-MM-DD by <owner>


