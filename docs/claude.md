# claude.md
Centralized context + instructions for using Claude in this repo.

## 🎯 Role
You are Claude, acting as an engineering assistant for **Aurora Trading System**.
Optimize for **small, deterministic, reversible** changes behind flags.

## 🔗 Inherit Aurora ADOI
Read `adoi.yaml` in the repo as the source of truth. Obey:
- **Prime directive:** ship small, deterministic, reversible changes behind flags.
- **Invariants:** no_data_leakage, deterministic_builds, reproducible_experiments_with_manifests, ci_gates_guard_contracts, golden_snapshot_for_smoke, export_parity_tests, purged_forward_splits_only, risk_profiles_single_source.
- **Definition of done:** code, tests, docs, CI green, artifacts manifest, rollback path.

## 🧭 Workflow (must follow)
1) **Spec** → problem, user impact, success metric, guardrails, flags, affected modules  
2) **Contract Gates** → schemas, labels contract, risk profile, API I/O, perf budgets  
3) **Implement** → one flagged path; smallest surface area; no rewrites  
4) **Test** → unit, contract/property, metamorphic, mutation, integration, golden smoke  
5) **Review** → PR template; CI must pass  
6) **Ship** → enable flags dev → staging → prod; release note + rollback note

## ✅ Guardrails (hard)
- All new behavior **disabled by default** (feature flags).
- Seeds fixed; **no nondeterminism**; purged forward splits for ML.
- Perf budgets: unit ≤1s, train‑smoke ≤60s, memory ≤2GB.
- **Parity tests** for exporters (onnx/native/torchscript/joblib) abs err ≤ 1e‑5.
- Artifacts manifest includes: code_sha, data_snapshot, featureset, trainer, exports, split_policy, random_seed.
- Rollback: **single env var/flag**.

## 📦 Expected Output Formats
For any change, produce these sections **in the reply**:

### 1) Patch
```diff
# minimal, surgical diff; one flagged path
```

### 2) Tests

* Unit: cases
* Contract/Schema checks: what & where
* Parity/Determinism: seed + tolerance
* Golden smoke: snapshot + runtime budget

### 3) Artifacts & CI

* Artifacts written + manifest delta
* CI additions/edits (lint/datasanity/train\_smoke/parity/integration)

### 4) Risk & Rollback

* Risk profile touched
* Rollback switch (flag/env) + command/runbook

### 5) Release Notes (1–2 lines)

* What changed
* User‑visible impact (if any)

## 🧪 DataSanity / ML specifics (if applicable)

* Input contracts: float64\_OHLCV, tz‑aware UTC, deduped index, monotonic timestamps.
* Label policy: explicit horizon/lag; side‑effects documented.
* Split policy: **purged forward splits only**; no peeking.

## 🔄 Example Prompts

* "Write unit + property tests for `src/features/returns.py` to catch identical‑price runs without lookahead; include golden smoke wiring."
* "Add ONNX export parity test for `golden_xgb_v2` with abs err ≤1e‑5; update CI."
* "Draft rollback runbook entry to disable `signals.v2` via `FLAG_SIGNALS_V2=0`."
