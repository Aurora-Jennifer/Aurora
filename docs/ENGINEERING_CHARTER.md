# AURORA ENGINEERING CHARTER

**Unified: Audit Trail + Refactor Safety + Development Discipline**

This charter defines *how* Aurora engineering operates.  
It complements the **Cursor Master Ruleset** (machine-enforced guardrails).

---

## 0. Operating Mode

1. **Safety-first workflow**
   - Read `config/guardrails.yaml` if present.
   - **Dry-run first**: plan only, no writes.
   - Wait for explicit approval token: `APPROVE: REPO-###`.
   - Apply **minimal, surgical diffs**.
   - Run verification commands.
   - Provide rollback instructions.

2. **Scope discipline**
   - Only touch files listed in the plan.
   - Never reformat unrelated lines.

3. **Change budgets (hard limits)**
   - Max files changed: 25  
   - Max insertions: 1200  
   - Max deletions: 600  
   - Max moves/renames: 5  

---

## 1. Path Policy

- **Protected (no edits without override):**

```

core/ runtime/ engine/ data\_sanity/ brokers/
signals/ strategies/ state/ runs/ results/
SECURITY\_CHECKLIST.md MASTER\_DOCUMENTATION.md
INVESTOR\_PRESENTATION.md PROVENANCE.sha256

```

- **Allowed (safe to modify):**

```

config/ scripts/ cli/ tests/ tools/ utils/ docs/
experiments/ attic/ features/ ml/ viz/ reports/
logs/ artifacts/

````

---

## 2. Audit Trail

**Always required** for any repo change.

- Create under `docs/audit/YYYY-MM-DD/HH-MM-SS_<topic>/`:

- `00_Roadmap.md` — intent, plan, success
- `10_Changes.md` — actions, commands
- `20_Diff.md` — files touched, line counts
- `30_Risks.md` — assumptions, rollback
- `40_TODO.md` — follow-ups

- Daily rollup: `docs/audit/YYYY-MM-DD/EOD.md`  
Append timeline entries like:  
`[HH:MM] <topic> → files:N tests:pass risk:low|med|high`

**Exceptions:** Only if user explicitly says “no audit trail” or request is informational only.

---

## 3. Engineering Directives

1. **Never index empties.** Guard every `[-1]`, `.iloc[-1]`. Empty → return `HOLD`.
2. **No magic numbers.** All runtime knobs come from `config/*.yaml`.
3. **Short folds.** If `test_len < step_size`: truncate if allowed, else skip with warning.
4. **Warmup discipline.** Require `min_history_bars`. Drop NAs, slice afterward.
5. **Log hygiene.** One fold-level summary per cause; no per-bar spam.
6. **yfinance explicitness.** Always pass `auto_adjust` from config.

---

## 4. Config & Loader

- **Base config**: `config/base.yaml` defines engine, walkforward, data, risk, composer, tickers.
- **Overlays**: profiles (`risk_low`, `risk_balanced`, `risk_strict`), tasks (`backtest`, `walkforward`, `paper`), assets (`equity`, `crypto`).
- **Loader API**:
```python
cfg = load_config([Path("config/base.yaml"), *overlays])
get_cfg(path: str, default: Any | None = None) -> Any
````

---

## 5. Data Sanity & Fold Builder

* `config/data_sanity.yaml` required with:

  * `enabled: true|false`
  * `mode: enforce|warn|off`
* Enforce mode with violations → graceful abort (no stack trace).
* Fold builder must implement short-fold policy.
* Unit tests must cover both truncate and skip branches.

---

## 6. Testing & CI

* **Tools**: `ruff`, `mypy/pyright`, `pytest`.
* **Markers**: `smoke`, `quarantine`.
* **Test structure**:

  ```
  tests/
    unit/
    integration/
    walkforward/
    e2e/
  ```
* **Must-have tests**:

  * Empty features → HOLD (no exception).
  * Short fold handling.
  * `min_history_bars` enforcement.
  * Config overlay precedence.
  * yfinance auto\_adjust propagation.

---

## 7. Observability & Metadata

* **JSON logs** per run with: `run_id`, `symbol`, `fold`, `latency_ms`, `pnl`.
* **Run header JSON** includes:

  ```json
  {
    "run_id": "YYYY-MM-DDTHH-MM-SSZ",
    "git_sha": "<short>",
    "config": "configs/<file>.yaml",
    "data_snapshot": "s3://.../YYYY-MM-DD",
    "seed": 1337
  }
  ```

---

## 8. Ergonomics

Provide one-command tasks in `Justfile` or `Makefile`:

```
just test       # lint + type + unit
just integ      # integration
just backtest   # run backtest & write reports/run.json
just paper      # start paper runner
just promote    # validate metrics & tag release
just index      # tools/repo_index.py
```

---

## 9. Documentation & Summaries

* Auto-generate `docs/summaries/<folder>.md` with purpose, entrypoints, APIs.
* Root `MODULE_MAP.md` for quick repo orientation.

---

## 10. Commit & Rollback

* **Commit template:**

  ```
  feat(<area>): <concise change>

  - Summary of change (why)
  - Touched: paths (+N/−M)
  - Tests: <paths> (pass)
  - Perf: runtime ±X% on <dataset>
  - Audit: docs/audit/YYYY-MM-DD/HH-MM-SS_<topic>/
  ```

* **Rollback**: `git checkout -p <paths>` or revert tag.

---

## 11. Minimal Audit Templates

See [docs/audit/templates](../docs/audit/) for 00\_..40\_ templates and `EOD.md` format.
*Each change must create/update them.*

---

## 12. Definition of Done

* No unguarded tail indexing.
* No magic numbers in source.
* Tests pass; new behavior covered.
* Logs clean; fold summaries only.
* README/CHANGELOG updated if behavior changes.
* Audit trail created + EOD updated.

---

## 13. Usage

1. Run audit dry-run plan.
2. Approve with token: `APPROVE: REPO-###`.
3. Apply minimal diffs.
4. Verify & commit with audit references.
5. Update `EOD.md`.

---

**End of Charter**

```

[Next]  
Drop this into `docs/ENGINEERING_CHARTER.md`. Cursor enforces the compressed ruleset; humans follow this charter.  

Do you want me to also generate the **audit file skeletons** (`00_Roadmap.md`, etc.) pre-created under `docs/audit/templates/` so they’re boilerplate-ready?
```
