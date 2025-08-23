# Risks & Assumptions
- **Assumption**: Current logging infrastructure can be extended for metrics
- **Assumption**: Performance overhead of metrics collection is acceptable
- **Risk**: Metrics collection may add latency to trading loop
- **Risk**: Memory monitoring may impact performance
- **Rollback**: `FLAG_COMPREHENSIVE_METRICS=0` disables new metrics
- **Rollback**: Remove metrics collection from trading loop if issues arise
