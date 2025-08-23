# Risks & Assumptions
- **Assumption**: Existing golden data in tests/golden/ is stable and representative
- **Assumption**: E2E pipeline can be modified to use frozen snapshot flag
- **Risk**: Frozen snapshot may become stale over time
- **Risk**: CI validation step may add runtime overhead
- **Rollback**: `FLAG_GOLDEN_SNAPSHOT_FROZEN=0` reverts to live snapshot
- **Rollback**: Remove CI snapshot validation step if issues arise
