## Core System Files and Directories (Keep List)

This document lists the minimal set of files and directories to keep active for day-to-day operation, debugging, and continued alpha/model development. Everything else can be archived under `attic/` and restored if needed.

- Paper Trading Automation
  - daily_paper_trading.sh
  - monitor_paper_trading.sh
  - ops/daily_paper_trading.py
  - ops/enhanced_dry_run.py
  - systemd/paper-trading.service
  - systemd/paper-trading.timer
  - ~/.config/systemd/user/paper-preflight.service (generated)
  - ~/.config/systemd/user/paper-preflight.timer (generated)
  - ~/.config/systemd/user/paper-eod.service (generated)
  - ~/.config/systemd/user/paper-eod.timer (generated)
  - ~/.config/systemd/user/paper-data-fetch.service (generated)
  - ~/.config/systemd/user/paper-data-fetch.timer (generated)
  - ~/.config/systemd/user/paper-status.service (generated)
  - ~/.config/systemd/user/paper-status.timer (generated)

- Data Pipeline
  - tools/fetch_bars_alpaca.py
  - data/universe/top300.txt
  - data/latest/ (prices.parquet, fundamentals.parquet)

- Feature Engineering and Integrity
  - ml/panel_builder.py
  - ml/leakage_audit.py
  - ml/structural_leakage_audit.py
  - ml/negative_controls.py
  - snapshots/sector_map.parquet

- Modeling & Evaluation
  - scripts/run_universe.py
  - ml/runner_universe.py
  - ml/metrics_market_neutral.py
  - ml/score_weight_mapping.py

- Risk, Costs, Capacity
  - ml/capacity_enforcement.py
  - ml/impact_model.py

- Ops Utilities
  - ops/pre_market_dry_run.py
  - ops/paper_trading_guards.py
  - ops/date_helpers.py
  - tools/ (general helpers kept as needed)

- Config & Environment (active)
  - ~/.config/paper-trading.env
  - config/base.yaml (if present)
  - config/data_sanity.yaml (if present)

- Documentation (active)
  - docs/ARCHITECTURE_OVERVIEW.md
  - docs/AUTOMATED_PAPER_TRADING_GUIDE.md
  - docs/DATA_PIPELINE_ARCHITECTURE.md
  - docs/LAUNCH_READINESS_CHECKLIST.md
  - docs/SYSTEMD_AUTOMATION_GUIDE.md
  - docs/CORE_SYSTEM_FILES.md (this file)

- Results & Logs (active outputs)
  - results/paper/
  - logs/systemd_preflight.log

Notes
- Keep `requirements*.txt` and `requirements.lock` for reproducible envs.
- Keep `pytest.ini`, `Makefile`, and CI workflow if present.
- Archive legacy/unused scripts, configs, and reports under `attic/`.

Suggested Archive Targets (safe to move if not referenced)
- attic/ (already archived content)
- experiments/, examples/, notebooks/, old scripts in scripts/
- old READMEs and presentations in root (now moved under attic/old_docs/)

How to Archive
- Create category under `attic/` and move non-core files:
  - mkdir -p attic/legacy_scripts && git mv scripts/old_*.py attic/legacy_scripts/ 2>/dev/null || true
  - mkdir -p attic/legacy_configs && git mv config/*.backup* attic/legacy_configs/ 2>/dev/null || true
- Commit with message: "chore(attic): archive non-core files (ref: docs/CORE_SYSTEM_FILES.md)"


- Requirements, Tooling, CI (keep)
  - requirements.txt, requirements-lock.txt
  - pyproject.toml (or setup.cfg)
  - Makefile, pytest.ini
  - .github/workflows/ci.yml, .gitignore, CONTRIBUTING.md (if present)

- Active Config & Artifacts (keep)
  - config/base.yaml (and overlays actually used)
  - config/data_sanity.yaml (only if enforced in CI/runtime)
  - snapshots/sector_map.parquet
  - data/universe/top300.txt

- Docs (keep)
  - README.md
  - docs/ARCHITECTURE_OVERVIEW.md
  - docs/AUTOMATED_PAPER_TRADING_GUIDE.md
  - docs/DATA_PIPELINE_ARCHITECTURE.md
  - docs/LAUNCH_READINESS_CHECKLIST.md
  - docs/SYSTEMD_AUTOMATION_GUIDE.md
  - docs/CORE_SYSTEM_FILES.md

- Outputs (keep directories; rotate contents)
  - results/paper/
  - logs/

Data Sanity Layer — Is it outdated?
- Current production flow relies on: enhanced dry-run coverage gates, leakage audits, whitelist validation, sector snapshot checks, and risk guards.
- Legacy DataSanity modules exist under `core/data_sanity/**` and `src/aurora/data_sanity/**`.
- Recommendation:
  - If you do NOT call DataSanity in CI/runtime: treat the legacy DataSanity tree as archival; keep only `config/data_sanity.yaml` if referenced by tests; otherwise archive it with the rest under `attic/`.
  - If you still run DataSanity tests in CI: keep `core/data_sanity/api.py`, `core/data_sanity/main.py`, and `config/data_sanity.yaml`; archive the rest not referenced by imports or tests.
- Action item: decide per CI usage. I can scan CI/tests to confirm references and update this file accordingly on request.
