# Changes
## Actions
- docs/audit/2025-08-22/10-00_export_parity/00_Roadmap.md: add — intent/plan/success
- docs/audit/2025-08-22/10-00_export_parity/10_Changes.md: add — actions/commands
- docs/audit/2025-08-22/10-00_export_parity/20_Diff.md: add — files touched + counts
- docs/audit/2025-08-22/10-00_export_parity/30_Risks.md: add — assumptions/rollback
- docs/audit/2025-08-22/10-00_export_parity/40_TODO.md: add — follow-ups
- docs/audit/2025-08-22/EOD.md: update — timeline entry

## Commands run
```bash
# Create audit structure
mkdir -p docs/audit/2025-08-22/10-00_export_parity/

# Test existing ONNX parity implementation
python scripts/onnx_parity.py --atol 1e-5

# Verify parity results
cat reports/experiments/parity.json
```
