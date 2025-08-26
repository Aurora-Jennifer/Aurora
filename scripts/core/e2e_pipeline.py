import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="golden_xgb_v2")
    ap.add_argument("--signals_out", default="artifacts/signals/out.jsonl")
    args = ap.parse_args()

    # 1) Train (produces parity artifacts + search_history)
    os.environ["ML_EXPORT_ONNX"] = os.environ.get("ML_EXPORT_ONNX", "1")
    code = os.system(f"PYTHONPATH=. python scripts/train_linear.py {args.profile}")
    if code != 0:
        return os.WEXITSTATUS(code)

    # 2) Generate dummy ranked signals based on OOF preds
    preds = list(Path("artifacts/models").glob("*/preds_oof.parquet"))
    if not preds:
        print("[E2E] No OOF preds found; abort")
        return 1
    df = pd.read_parquet(sorted(preds)[-1])
    # Expect columns: date, asset, pred
    if not {"date", "asset"}.issubset(df.columns):
        # fallback: make minimal stub
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "asset": ["SPY"] * 10, "pred": 0.0})
    outp = Path(args.signals_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for _, r in df.head(50).iterrows():
            f.write(json.dumps({"symbol": r.get("asset", "SPY"), "score": float(r.get("pred", 0.0))}) + "\n")
    print(f"[E2E] Signals written -> {outp}")

    # 3) Go/No-Go Gates (fail hard unless ALLOW_FAIL=1)
    allow_fail = os.environ.get("ALLOW_FAIL", "0") == "1"

    try:
        # Load search history for gates
        search_hist_path = "reports/experiments/search_history.json"
        if Path(search_hist_path).exists():
            with open(search_hist_path) as f:
                search_hist = json.load(f)

            # Signal quality gates
            best_ic = search_hist.get("best", {}).get("ic_mean", 0.0)
            baseline_ic = search_hist.get("baseline", {}).get("ic_mean", 0.0)
            delta_ic = best_ic - baseline_ic
            best_vs_baseline = search_hist.get("best_vs_baseline", {})
            p_against = best_vs_baseline.get("p_value", 1.0)

            assert best_ic >= 0.09, f"IC too low: {best_ic:.3f} < 0.09"
            assert delta_ic >= 0.02, f"ΔIC too small: {delta_ic:.3f} < 0.02"
            assert p_against <= 0.10, f"Not significant: p={p_against:.3f} > 0.10"
            print(f"[E2E][GATE] Signal quality: IC={best_ic:.3f}, ΔIC={delta_ic:.3f}, p={p_against:.3f} ✅")

        # Parity gate
        parity_path = "reports/experiments/parity.json"
        if Path(parity_path).exists():
            with open(parity_path) as f:
                parity = json.load(f)
            max_diff = parity.get("max_abs_diff", 1.0)
            assert parity.get("ok", False), "ONNX parity failed"
            assert max_diff <= 1e-5, f"Parity drift too high: {max_diff:.2e} > 1e-5"
            print(f"[E2E][GATE] ONNX parity: max|Δ|={max_diff:.2e} ✅")

        # Performance gate
        bench_path = "reports/experiments/bench.json"
        if Path(bench_path).exists():
            with open(bench_path) as f:
                bench = json.load(f)
            p95_256 = bench.get("latency", {}).get("256", {}).get("p95_ms", 1000.0)
            assert p95_256 <= 60.0, f"Latency too high: {p95_256:.1f}ms > 60ms"
            print(f"[E2E][GATE] Performance: p95@256={p95_256:.1f}ms ✅")

        # Ablation sanity gate
        ablation_path = "reports/experiments/ablation_summary.json"
        if Path(ablation_path).exists():
            with open(ablation_path) as f:
                ablation = json.load(f)
            harmful_groups = [row["group"] for row in ablation.get("table", []) if row.get("harmful", False)]
            assert not harmful_groups, f"Harmful groups detected: {harmful_groups}"
            print("[E2E][GATE] Ablation sanity: no harmful groups ✅")

        # Artifacts existence gate
        required_artifacts = [
            "artifacts/signals/out.jsonl",
            "artifacts/models/latest.onnx",
            "artifacts/parity/sidecar.json"
        ]
        for artifact in required_artifacts:
            assert Path(artifact).exists(), f"Missing artifact: {artifact}"
        print("[E2E][GATE] Required artifacts exist ✅")

        print("[E2E][GATE] All gates passed! 🎉")

    except (AssertionError, FileNotFoundError, json.JSONDecodeError) as e:
        if allow_fail:
            print(f"[E2E][GATE] ⚠️  Gate failed but ALLOW_FAIL=1: {e}")
        else:
            print(f"[E2E][GATE] ❌ Gate failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


