# Task Cards Examples
Example task cards following the ADOI-aware workflow.

---

### TITLE
Fix guard hash nondeterminism in DataSanity

**Status:** [ ] todo  
**Flag:** FLAG_SANITY_GUARD_HASH_V2=0  
**Owner:** Jennifer • **ETA:** 2025‑08‑23 • **WIP slot:** Today

#### 1) SPEC
- **Problem:** DataSanityGuard uses `hash()` which is salted → nondeterministic across runs
- **Impact:** CI becomes flaky; artifacts can't be reproduced consistently
- **Success Metric:** Guard hash identical across runs with same data; E2E determinism check passes
- **Guardrails touched:** deterministic_builds, reproducible_experiments_with_manifests

#### 2) CONTRACT GATES
- No schema change; API unchanged; perf unchanged
- Hash must be stable for identical DataFrame content

#### 3) IMPLEMENT
- Replace `hash(str(clean_data.values.tobytes()))` with `sha256(bytes).hexdigest()[:8]`
- Use stable content: shape + dtypes + head/tail sample
- Add `rebind(df)` method instead of mutating guard internals

#### 4) TEST PLAN
- **Unit:** identical DataFrames produce identical hashes across runs
- **Property:** hash changes when data changes, stays same when data identical
- **Determinism:** run E2E twice with same seed, verify identical artifacts
- **Golden Smoke:** `make e2e` ≤60s, hash stable

#### 5) CI & ARTIFACTS
- Add `tests/test_datasanity_guard_hash.py`
- Manifest: code_sha updated when guard logic changes

#### 6) RISK & ROLLBACK
- Low risk. Rollback: `FLAG_SANITY_GUARD_HASH_V2=0`
- Verify: run E2E twice, check hash consistency

#### 7) RELEASE NOTE
- Fixed DataSanity guard hash to be deterministic across runs; eliminates CI flakiness.

---

### TITLE
Unify validator engines (prevent inconsistent behavior)

**Status:** [ ] todo  
**Flag:** FLAG_SANITY_UNIFIED_VALIDATOR=0  
**Owner:** Jennifer • **ETA:** 2025‑08‑24 • **WIP slot:** Today

#### 1) SPEC
- **Problem:** Two overlapping validators (strict helpers + non-staged paths) = inconsistent behavior
- **Impact:** Fixes can land in one but not the other; unpredictable validation results
- **Success Metric:** Single validation engine; all paths use same logic; 0 regressions
- **Guardrails touched:** ci_gates_guard_contracts, no_data_leakage

#### 2) CONTRACT GATES
- Keep staged pipeline (canonicalize → coerce → invariants → rules → volume → outliers → returns → final)
- Delete duplicate non-staged helpers
- Maintain backward compatibility for existing configs

#### 3) IMPLEMENT
- Deprecate `_validate_*_strict` helpers
- Route all validation through staged pipeline
- Add deprecation warnings for old paths
- Ensure all rules registered in staged pipeline

#### 4) TEST PLAN
- **Unit:** all existing validation tests pass
- **Integration:** staged vs non-staged produce identical results
- **Golden Smoke:** E2E produces identical outputs
- **Regression:** no new validation failures on existing data

#### 5) CI & ARTIFACTS
- Update CI to use unified validator
- Add deprecation test warnings
- Manifest: validation_engine_version field

#### 6) RISK & ROLLBACK
- Medium risk (touches core validation). Rollback: `FLAG_SANITY_UNIFIED_VALIDATOR=0`
- Verify: run full test suite, check E2E outputs

#### 7) RELEASE NOTE
- Unified DataSanity validation engine; eliminates inconsistent behavior between strict/non-strict paths.

---

### TITLE
Make strict profile actually strict

**Status:** [ ] todo  
**Flag:** FLAG_SANITY_STRICT_MODE=0  
**Owner:** Jennifer • **ETA:** 2025‑08‑25 • **WIP slot:** Today

#### 1) SPEC
- **Problem:** Strict profile allows repairs even when configured to fail-fast
- **Impact:** Data quality issues masked; upstream bugs hidden
- **Success Metric:** Strict mode fails fast on violations; no silent repairs
- **Guardrails touched:** ci_gates_guard_contracts, no_data_leakage

#### 2) CONTRACT GATES
- Add `mode: fail` config option
- Strict mode must raise `DataSanityError` for violations
- No repairs allowed in strict mode

#### 3) IMPLEMENT
- Add `mode: fail` to strict profile config
- Modify validation logic to respect mode setting
- Raise `DataSanityError` instead of repairing in strict mode
- Add clear error messages with violation details

#### 4) TEST PLAN
- **Unit:** strict mode raises on non-positive prices, extreme returns, etc.
- **Integration:** strict profile fails fast on golden data violations
- **Golden Smoke:** strict mode fails as expected, non-strict repairs
- **Regression:** existing non-strict behavior unchanged

#### 5) CI & ARTIFACTS
- Add strict mode tests to CI
- Update profile configs to include mode setting
- Manifest: validation_mode field

#### 6) RISK & ROLLBACK
- Medium risk (changes failure behavior). Rollback: `FLAG_SANITY_STRICT_MODE=0`
- Verify: strict mode fails appropriately, non-strict still works

#### 7) RELEASE NOTE
- Made strict profile actually strict; fails fast on data violations instead of silent repairs.

---

### TITLE
Add ONNX export parity test for golden_xgb_v2

**Status:** [ ] todo  
**Flag:** FLAG_EXPORT_PARITY=0  
**Owner:** Jennifer • **ETA:** 2025‑08‑26 • **WIP slot:** This Week

#### 1) SPEC
- **Problem:** No enforced parity guard; risk of silent export drift
- **Impact:** Prevents deployment of broken ONNX models
- **Success Metric:** abs err ≤ 1e‑5 vs native on golden features
- **Guardrails touched:** export_parity_tests, ci_gates_guard_contracts

#### 2) CONTRACT GATES
- ONNX output must match native within tolerance
- Test on golden snapshot features
- Fail CI if tolerance exceeded

#### 3) IMPLEMENT
- Create ONNX export parity test
- Compare predictions on golden features
- Add tolerance checking (abs err ≤ 1e‑5)
- Integrate into CI pipeline

#### 4) TEST PLAN
- **Unit:** small tensor cases; edge NaNs masked
- **Integration:** compare batched predictions on golden snapshot
- **Parity:** abs err ≤ 1e‑5 on all test cases
- **Golden Smoke:** export test runs ≤30s

#### 5) CI & ARTIFACTS
- New step: `export_parity` in CI; fail hard on tolerance breach
- Artifacts: ONNX model + parity test results

#### 6) RISK & ROLLBACK
- Medium risk. Rollback: `FLAG_EXPORT_PARITY=0`
- Verify: parity test passes, no deployment drift

#### 7) RELEASE NOTE
- Added ONNX export parity CI gate to prevent drift in deployments.
