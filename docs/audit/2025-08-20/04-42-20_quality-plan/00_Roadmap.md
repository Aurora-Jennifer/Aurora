# Post-CI Quality Plan — Traceability, Oracles, Teeth

## Goal
Introduce test discipline that proves tests hit target code paths, assert clear oracles, and fail when invariants are broken — without blocking CI.

## Scope (after CI hardening + smoke + promotion)
- Focus Tier-1 rules first (e.g., duplicates, no_lookahead).
- Minimal diffs; no core refactors.

## Phase 1 — Traceability (2–3h)
- Per-test coverage contexts: pytest --cov --cov-context=test --cov-report=json:coverage.json
- Tool: tools/traceability.py → emits tests/_traceability.md with columns:
  test_name | target_fn(s) | invariant | inputs | severity | oracle_type
- CI allow-fail step uploads the markdown; gate locally on Tier‑1 mapping completeness.

## Phase 2 — Oracles & Labeled Fixtures (3–4h seed)
- Add tests/golden/clean and tests/golden/violations + labels.json (row_id → [rule_ids]).
- Convert asserts to check structured codes/ValidationResult for one rule first.
- Keep fixtures tiny (≤2KB each).

## Phase 3 — Teeth (fail-on-purpose) (2–3h)
- Mutation spot-check via monkeypatch per rule (flip predicate/null-guard).
- Re-run only mapped tests (from traceability); they must fail; record kill-rate.

## Phase 4 — Boundary & Flake Control (3–4h)
- Edge cases: empty, single-row, dup ts, tz skew, NA bursts, extreme values.
- Rerun N=10 with fixed seeds; quarantine >1% flake (mark=quarantine).

## Phase 5 — Precision/Recall Report (2–3h)
- tools/test_audit.py computes TP/FP/FN per rule on labeled sets.
- Emit docs/audit/<DATE>/test_audit.md; gates: recall ≥0.95, precision ≥0.98 (Tier‑1).

## CI additions (allow-fail)
- Generate tests/_traceability.md and upload artifact.
- Run tools/test_audit.py; upload short report; no blocking yet.

## Exit gates (Tier‑1)
- 100%% mapped tests hit target functions (traceability table).
- Clean + ≥5 violating labeled rows per rule with specific error codes.
- Mutation kill-rate ≥95%%; boundary coverage ≥90%%; flake ≤1%%.
- Precision ≥0.98, Recall ≥0.95 on labeled sets.

## Deliverables
- tools/traceability.py, tools/test_audit.py
- tests/_traceability.md (generated)
- tests/golden/{clean,violations}/ + labels.json
- One mutation test per Tier‑1 rule (monkeypatch).

## Quick commands
- pytest -q --cov=. --cov-context=test --cov-report=json:coverage.json
- python tools/traceability.py > tests/_traceability.md
- pytest -q -k "tier1_rule"
- python tools/test_audit.py --golden tests/golden > docs/audit/2025-08-20/test_audit.md
