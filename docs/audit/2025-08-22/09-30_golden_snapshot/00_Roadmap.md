# Golden Snapshot Freeze — Roadmap (2025-08-22 09:30)
**Prompt:** "Freeze golden snapshot (deterministic reference dataset)"

## Context
- Paper trading readiness checklist shows 66% complete
- Golden snapshot exists but not formally frozen/versioned
- Need deterministic reference dataset for reproducible experiments
- Current TODO shows this as highest priority item

## Plan (now)
1) Create `artifacts/snapshots/golden_ml_v1_frozen/` directory structure
2) Generate hash manifest for existing golden data
3) Update E2E pipeline to use frozen snapshot when flag enabled
4) Add snapshot validation in CI
5) Update model manifest to include snapshot hash

## Success criteria
- Golden snapshot hash frozen and versioned
- E2E tests pass consistently with frozen snapshot
- CI includes snapshot validation step
- Model manifest includes snapshot hash reference
- Perf budget: snapshot load ≤5s maintained
