# Risks

## Assumptions
- Current DataSanity v1 behavior is stable and correct
- Configuration files are accessible and readable
- Telemetry output directory is writable
- v2 engine will be implemented as an enhancement, not a rewrite

## Potential Issues
1. **Performance overhead**: Telemetry and metrics collection may add latency to validation
2. **Storage growth**: JSONL telemetry files may grow large over time
3. **Configuration complexity**: Multiple config sources may lead to confusion
4. **Engine switch bugs**: Routing logic may have edge cases not covered by tests
5. **Telemetry data loss**: JSONL files may be lost or corrupted

## Mitigation
- **Performance**: Telemetry is configurable and can be disabled if needed
- **Storage**: Implement log rotation and cleanup policies
- **Configuration**: Centralized config access with clear precedence rules
- **Engine switch**: Comprehensive test coverage and fallback to v1
- **Data loss**: Telemetry is non-critical for core functionality

## Rollback
- **Immediate**: Set `datasanity.engine: v1` in config to revert to v1
- **Telemetry**: Disable with `datasanity.telemetry.enabled: false`
- **Pre-commit**: Remove DataSanity hook from `.pre-commit-config.yaml`
- **Canary**: Stop running canary tests if they cause issues

## Monitoring
- Watch for performance degradation in validation times
- Monitor telemetry file sizes and cleanup regularly
- Track canary test results for regressions
- Alert on v2-only failures in canary tests
