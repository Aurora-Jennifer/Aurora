# Risks & Assumptions
- **Assumption**: ONNX exports are working correctly for golden_xgb_v2 model
- **Assumption**: Native model predictions are the ground truth
- **Risk**: Parity test may be too strict and fail on legitimate numerical differences
- **Risk**: CI integration may add runtime overhead
- **Rollback**: `FLAG_EXPORT_PARITY_TESTS=0` disables parity checks
- **Rollback**: Remove CI parity validation step if issues arise
