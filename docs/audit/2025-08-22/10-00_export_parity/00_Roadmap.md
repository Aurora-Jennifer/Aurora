# Export Parity Tests — Roadmap (2025-08-22 10:00)
**Prompt:** "Add export parity tests (prevent silent model drift)"

## Context
- Paper trading readiness checklist shows export parity tests as next priority
- Need to prevent silent model drift between training and deployment
- ONNX exports working but no automated parity tests
- Golden snapshot freeze completed successfully

## Plan (now)
1) Create export parity test script comparing ONNX vs native predictions
2) Wire to CI with tolerance checking (abs err ≤ 1e-5)
3) Test on golden_xgb_v2 model
4) Add CI integration with failure on tolerance breach
5) Create parity test artifacts and reporting

## Success criteria
- Export parity test script validates ONNX vs native predictions
- CI includes parity validation step
- Tolerance enforced: abs err ≤ 1e-5 for float outputs
- Test runs on golden_xgb_v2 model successfully
- Parity test results logged and artifacts created
