# Aurora Guard Index

Single source of truth for all Cursor rulesets and their intent. If a rule isn’t listed here, it’s unofficial.

## Table of Contents

| Guard Name                      | File                                  | Scope Summary |
|---------------------------------|---------------------------------------|---------------|
| CI/CD Guard                     | `.cursor/guards/ci_cd_guard.yaml`     | Smoke-only blocking, allow-fail full tests + lint, pre-push smoke, badge |
| Unified Repo Guard              | `.cursor/guards/unified_repo_guard.yaml` | Master budgets, path policy, audit trail, invariants |
| Test & Runtime Guard            | `.cursor/guards/test_runtime_guard.yaml` | Golden datasets, flaky policy, runtime timeouts |
| Security & Compliance Guard     | `.cursor/guards/security_guard.yaml`  | Bandit/Semgrep in CI, pinned deps, SBOM, secret patterns |
| Observability Guard             | `.cursor/guards/observability_guard.yaml` | Metrics schema, anomaly warnings, drift checks |
| Release & Deployment Guard      | `.cursor/guards/release_guard.yaml`   | Changelog + EOD on tag, build hash, promote gate, migrations on changes |
| Quant & Trading Guard           | `.cursor/guards/trading_guard.yaml`   | Exposure ≤100%, leverage cap, slippage/fees applied, walkforward fairness |
| Ergonomics & Process Guard      | `.cursor/guards/ergonomics_guard.yaml`| PR size ≤500 LOC, just smoke <60s, pre-push test hook, attic policy |
| Governance & Secrets Guard      | `.cursor/guards/governance_guard.yaml`| Secret patterns, licensing notes, provenance + PROVENANCE.sha256 |
| Resilience & Fail-Safe Guard    | `.cursor/guards/resilience_guard.yaml`| Kill-switch, rate-limit, fail-closed, emergency logging |
| Documentation & Knowledge Guard | `.cursor/guards/docs_guard.yaml`      | READMEs with Purpose/Entrypoints/Do-not-touch, Mermaid diagrams, runbooks |
| Data Integrity Guard            | `.cursor/guards/data_integrity_guard.yaml` | Schema contracts, NA policy, lineage in run.json, archival/index |
| Architecture & Dependency Guard | `.cursor/guards/architecture_guard.yaml` | Layer boundaries, API docs, deprecation warnings, lock freshness |
| CLEARFRAME Reasoning Guard      | `.cursor/guards/clearframe_guard.yaml`| Reasoner persona, contracts-first, coder handoff, enforcement hooks |

> Folder convention: keep all rulesets under `.cursor/guards/` and reference them from Cursor’s project config.

---

## Cross-Guard Contracts (must stay consistent)

- **Audit trail paths**: `docs/audit/YYYY-MM-DD/HH-MM-SS_<topic>/` + `EOD.md`
- **Reports**: `reports/run.json`, `reports/folds/*.json`, `reports/sbom.json`
- **Configs**: `config/base.yaml`, overlays in `config/**`, `config/data_schema.yaml`
- **CI**: `.github/workflows/ci.yml` with jobs: `smoke` (required), `lint` (allow-fail), `tests-full` (allow-fail), `security`
- **Kill-switch**: `kill.flag` polling in runner; test exists
- **Secrets**: reject AWS/OpenAI/GitHub/private key patterns repo-wide

---

## Drift Watchlist (common duplication hotspots)

- **Migration docs**: referenced by Release Guard *and* Ergonomics Guard — keep one owner (Release).
- **Audit TODO usage**: referenced by Security, Observability, Trading — unify phrasing: `docs/audit/**/40_TODO.md`.
- **Timeouts**: declared in Test & Runtime Guard and CI/CD Guard — ensure both specify identical budgets.

---

## Quick Verification (human checks)

- **Are all guards loaded?** Cursor → Project Rules: verify 13 entries match this table.
- **Any missing files?** `ls .cursor/guards/*.yaml` vs table above.
- **Schema keys stable?** `jq` keys in `reports/run.json` and `reports/folds/*.json` match Observability/Data Integrity.

---

## How to Add a New Guard (2 steps)

1) Create `.cursor/guards/<name>_guard.yaml` with:  
   - front-matter (`description`, `globs`, `alwaysApply: true`)  
   - enforceable rules (must-have / reject patterns)  
   - zero prose “policy” without checks

2) Append a new row to the table above with a one-line scope summary.

---

## Ownership

- **Maintainer:** you (repo owner)  
- **Change policy:** PR must include a short **Narrative** in `docs/audit/**/00_Roadmap.md` and link to the guard diff.

