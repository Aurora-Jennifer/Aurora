# DataSanity Production Rollout — Roadmap (2025-08-20 10:00)

**Prompt:** "Implement DataSanity production rollout foundation with engine switch, telemetry, and observability"

## Context
- DataSanity test suite is now stable with comprehensive validation
- Need production-ready infrastructure for safe rollout of stricter validation
- Goal: Enable v1/v2 side-by-side comparison with observability and kill-switches

## Plan (implemented)
1. **Engine Switch Facade** - Created `validate_and_repair_with_engine_switch()` function that routes to v1/v2 based on config
2. **Configuration Management** - Added `datasanity.engine` config key and centralized config access
3. **Telemetry System** - Implemented JSONL logging for validation runs with structured metrics
4. **Metrics Collection** - Added counters for validation checks, error codes, and severity levels
5. **Canary Testing** - Created script to compare v1 vs v2 outcomes on fixed corpus
6. **Pre-commit Integration** - Added DataSanity smoke test to pre-commit hooks
7. **Test Coverage** - Added comprehensive tests for engine switch functionality

## Success criteria
- [x] Engine switch correctly routes validation requests
- [x] Telemetry emits structured JSON logs for each validation run
- [x] Metrics track validation behavior and error budgets
- [x] Canary script compares v1 vs v2 and detects regressions
- [x] All tests pass with proper error handling
- [x] Configuration is centralized and accessible

## Results
- **Files changed**: 8 files
- **Key components**:
  - Engine switch facade with config-based routing
  - Telemetry system with JSONL output
  - Metrics collection with error budget tracking
  - Canary testing script with regression detection
  - Pre-commit integration for smoke tests
  - Comprehensive test coverage

## Next Steps
- Monitor canary test results in CI
- Implement v2 engine improvements behind the facade
- Add performance monitoring and alerting
- Expand telemetry to include more detailed metrics
- Create runbooks for operational procedures
