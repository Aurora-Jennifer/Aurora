## CI Decisions: Promotion Rules and Budgets

This document records promotion criteria and flip points for CI gates.

### ONNX Parity (blocking flip)
- Rule: Flip parity to blocking after 7 consecutive nightly greens on main.
- Action: Remove `|| echo "[PARITY] allow-fail"` from the parity step when threshold is met.

### Advisory Gates (remain non-blocking for now)
- Ablations: Warn if any group removal improves IC by > 0.005 with p ≤ 0.10.
- Significance: Print ΔIC and p; run nightly using `--use-after-costs`.
- Latency Bench: Track p50/p95; budget p95@32 ≤ 15–25 ms (CPU).
- Promotion: Summarize p95@32, IC_after_costs, ΔIC and ΔIC_after_costs.

### Budgets (initial)
- Latency p95@32: 15 ms (advisory)
- IC_after_costs floor: 0.05 (advisory)
- ΔIC_after_costs: ≥ 0.00 (advisory)

### Notes
- Keep new checks advisory for ~1 week before promotion.
- Prefer flags and profiles for behavior changes; do not delete legacy paths.


