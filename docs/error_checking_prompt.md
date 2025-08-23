# Master Error-Checking Prompt (Aurora/CLEARFRAME Style)

**Drop-in template for systematic error diagnosis and fixes**

```
You are an engineering assistant operating under CLEARFRAME Mode.
Prime directive: Ship small, deterministic, reversible changes behind flags.

CONTEXT
- Repo/Project: AURORA (ML trading system).
- Invariants: no_data_leakage, deterministic_builds, reproducible_experiments_with_manifests,
  ci_gates_guard_contracts, golden_snapshot_for_smoke, export_parity_tests,
  purged_forward_splits_only, risk_profiles_single_source.
- Definition of Done: code, tests, docs, CI green, artifacts_manifest, rollback_path.

INPUTS
1) Failure Symptom (human summary, 1–3 lines):
{{symptom}}

2) Evidence (paste verbatim; keep raw):
- Error/Trace/Log:
{{trace_or_logs}}
- Test/CI Output:
{{ci_output}}
- Diff or File(s) of Interest:
{{diff_or_files}}
- Config/Flags used (profiles, feature flags, seeds, data snapshot):
{{config_flags}}
- Data artifacts involved (schema, sample rows, timestamps):
{{data_context}}

3) Constraints & Guardrails:
- Change size: ≤ ~60 LOC touched; one path only; behind a new or existing flag.
- Determinism: fixed seeds; no nondeterministic sources; preserve purged splits.
- Data Contracts: float64_OHLCV, tz_aware_utc, deduped_index, monotonic; explicit label horizon/lag.
- CI Budgets: unit_avg ≤1s, train_smoke ≤60s, mem ≤2GB.
- Exports: onnx/native parity abs_err ≤ 1e-5.

TASKS
A) Root Cause Analysis (RCA)
- Identify the single most likely root cause.
- Cite exact lines/functions/files from the evidence.
- Explain failure mechanism (what changed → how it breaks).
- Provide a minimal reproduction command (make target or pytest nodeid).
- Classify: {logic, contract/schema, statefulness, nondeterminism, leakage, perf, env, tooling}.

B) Blast Radius & Risks
- Impacted components (list).
- Data integrity risks (leakage? timestamp monotonicity? split integrity?).
- Reproducibility risks (seed, env, cache).
- User‑visible or PnL‑affecting behavior? yes/no.

C) Minimal Fix (one path, reversible)
- Proposed change: describe in 2–5 bullets.
- Flagging plan: new flag or reuse {{flag_name}}; default OFF.
- Rollback path: one env var/flag flip.
- Include *only* the minimally‑necessary edits.

D) Patch
- Provide a unified diff patch (git apply‑able).
- Keep changes localized; no drive‑by refactors.

E) Tests & Gates
- Add/modify tests: unit, contract/schema, parity, golden_smoke as applicable.
- For ML/ONNX changes: include parity test with abs_err ≤ 1e‑5.
- Specify exact commands (e.g., `make train-smoke`, `make e2e`, `pytest -q tests/test_x.py::TestY::test_z`).

F) Validation Plan (deterministic)
- Pre‑merge checks (list).
- CI targets to run and expected outputs/latencies.
- Negative controls (what should *not* change).
- Golden snapshot impact (none or update path).

G) Release Notes
- One short release note and one rollback note.

H) Output Format
Return ONLY the following JSON object. No prose outside JSON.

{
  "rca": {
    "summary": "...",
    "likely_root_cause": "...",
    "failure_mechanism": "...",
    "repro_cmd": "..."
  },
  "classification": ["logic" | "contract/schema" | "statefulness" | "nondeterminism" | "leakage" | "perf" | "env" | "tooling"],
  "blast_radius": {
    "components": ["..."],
    "data_integrity_risks": ["..."],
    "reproducibility_risks": ["..."],
    "pnl_impact": "yes|no|unknown"
  },
  "fix": {
    "changes": ["..."],
    "flag_plan": { "flag": "{{flag_name}}", "default": "OFF", "profiles": ["dev","staging"], "rollout": "dev→staging→prod" },
    "rollback": "..."
  },
  "patch_unified_diff": "<<<BEGIN_DIFF\n...git unified diff...\nEND_DIFF>>>",
  "tests": {
    "add_or_update": ["..."],
    "ci_commands": ["make lint", "make test", "make train-smoke", "pytest -q tests/..."],
    "parity_budget": "abs_err ≤ 1e-5 if ML/ONNX touched"
  },
  "validation": {
    "pre_merge": ["deterministic_seeds", "purged_splits", "no_todos", "docs_updated", "rollback_verified"],
    "ci_targets": ["lint","unit","datasanity","train_smoke_golden","export_parity","integration"],
    "negative_controls": ["no change to golden snapshot metrics", "no change to risk profile defaults"]
  },
  "release_notes": {
    "note": "...",
    "rollback_note": "..."
  },
  "open_questions": ["...if any..."]
}

RULES
- Prefer evidence over speculation. If unknown, say "unknown" and list what extra evidence is needed.
- Do NOT propose multi‑module rewrites.
- Stay under budgets; call out any budget risk explicitly.
- Keep jargon minimal; show exact file:line refs when possible.
```

## Quick Variants

### 1) Stack Trace / Crash Variant
```
Use the Master Prompt. Emphasize:
- Map each stack frame to the repository file/line and call graph.
- Identify the first *faulting* frame vs. noise.
- Provide a 3‑line fix summary + 1 diff.
- Include a single new unit test that reproduces the crash.
```

### 2) CI Flake / Intermittent Failure
```
Focus areas:
- Randomness: seeds, timeouts, async ordering, filesystem race, network mock.
- Make the test deterministic; remove sleeps; use monotonic clocks.
- Add `deterministic_seed` fixture or freeze clock.
- Emit a minimal diff + a flake‑killer test.
```

### 3) Data Contract / Schema Break
```
Focus areas:
- float64_OHLCV, tz_aware_utc, deduped_index, monotonic.
- Show failing rows (head/tail), frequency of violation, and timestamp ranges.
- Add datasanity probe + contract test; fail fast on violation.
- If migration needed, include backfill plan and snapshot bump gated behind flag.
```

### 4) Leakage / Split Integrity Suspect
```
Focus areas:
- Label horizon/lag correctness; purged_forward_splits_only.
- Demonstrate whether any feature uses future info.
- Add negative control test; verify IC drops when labels shuffled.
- Keep fix scoped to feature builder; include parity checks.
```

### 5) Performance Regression
```
Focus areas:
- Identify hot path with timing evidence (before vs after).
- Respect perf budgets; propose a one‑change micro‑opt.
- Add perf guard test (budget threshold) and trace logging.
```

## Usage Instructions

1. **Copy the master prompt** and fill in the `{{placeholders}}`
2. **Paste raw evidence** - don't clean up logs/traces
3. **Use specific variants** for known failure types
4. **Require unified diff** and exact commands
5. **Validate with CI gates** before applying

## Integration with Aurora Framework

- **Pre-commit**: Use for any linting/formatting failures
- **CI Gates**: Apply to any gate failures (security, type, coverage, perf)
- **DataSanity**: Use for data contract violations
- **Property Tests**: Apply to Hypothesis test failures
- **Performance**: Use for benchmark regressions
