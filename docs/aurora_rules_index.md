# Aurora Guard Index

Single source of truth for all Cursor rulesets and their intent. If a rule isn’t listed here, it’s unofficial.

## Table of Contents

| Guard Name                      | File                                  | Scope Summary |
|---------------------------------|---------------------------------------|---------------|
# Aurora Guard Index

Single source of truth for all Cursor rulesets and their intent. If a rule isn’t listed here, it’s unofficial.

## Table of Contents

| Guard Name                      | File                                  | Scope Summary |
|---------------------------------|---------------------------------------|---------------|
| CI/CD Guard                     | `.github/workflows/ci.yml`            | Smoke-only blocking CI, allow-fail full tests + lint, pre-push smoke, badge |
| Unified Repo Guard              | `docs/ENGINEERING_CHARTER.md`         | Budgets, path policy, audit trail, invariants |
| Test & Runtime Guard            | `pytest.ini`                          | Markers, timeouts, quarantine policy |
| Security & Compliance Guard     | `.github/workflows/ci.yml`            | Bandit/Semgrep job, pinned deps, SBOM, secret patterns |
| Observability Guard             | `reports/metrics.schema.json`         | Metrics schema, anomaly warnings, drift checks |
| Release & Deployment Guard      | `.github/workflows/ci.yml`            | Changelog + EOD on tag, build hash, promote gate, migrations on changes |
| Quant & Trading Guard           | `config/base.yaml`                    | Exposure ≤100%, leverage cap, slippage/fees applied, walkforward fairness |
| Ergonomics & Process Guard      | `Makefile`                            | just/make smoke, pre-push smoke, PR size hints, attic policy |
| Governance & Secrets Guard      | `.gitleaks.toml` (optional)           | Secret patterns, licensing notes, provenance |
| Resilience & Fail-Safe Guard    | `scripts/*` (kill flag), tests        | Kill-switch, rate-limit, fail-closed, emergency logging |
| Documentation & Knowledge Guard | `docs/README.md`                      | READMEs with Purpose/Entrypoints/Do-not-touch, Mermaid diagrams, runbooks |
| Data Integrity Guard            | `config/data_schema.yaml`             | Schema contracts, NA policy, lineage in run.json, archival/index |
| Architecture & Dependency Guard | `docs/summaries/core.md`              | Layer boundaries, API docs, deprecation warnings, lock freshness |

> Pointers:
> - DataSanity: `docs/summaries/data_sanity.md`, `docs/runbooks/datasanity_ops.md`
> - Attic policy: `attic/docs/README.md`

---

## Cross-Guard Contracts (must stay consistent)

- Audit trail: `docs/audit/YYYY-MM-DD/HH-MM-SS_<topic>/` and `docs/audit/YYYY-MM-DD/EOD.md`
- Reports: `reports/run.json`, `reports/folds/*.json`, optional `reports/sbom.json`
- Configs: `config/base.yaml`, overlays in `config/**`, `config/data_schema.yaml`
- CI: `.github/workflows/ci.yml` with jobs: `smoke` (required), `lint` (allow-fail), `tests-full` (allow-fail), `security`
- Kill-switch: `kill.flag` polling and test
- Secrets: reject AWS/OpenAI/GitHub/private key patterns repo-wide

---

## Quick Verification (human checks)

- CI Smoke badge present in root README
- Pre-push smoke script present and executable
- Metrics schema file exists and analyzer runs (non-blocking)
- README stubs present in top-level modules (Purpose/Entrypoints/Do-not-touch)
.
- **Timeouts**: declared in Test & Runtime Guard and CI/CD Guard — ensure both specify identical budgets.

---

## Quick Verification (human checks)

- CI Smoke badge present in root README
- Pre-push smoke script present and executable
- Metrics schema file exists and analyzer runs (non-blocking)
- README stubs present in top-level modules (Purpose/Entrypoints/Do-not-touch)

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

