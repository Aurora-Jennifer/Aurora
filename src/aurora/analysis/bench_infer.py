#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _run(sess, X: np.ndarray, iters: int = 300, warmup: int = 50) -> tuple[float, float]:
    inp_name = sess.get_inputs()[0].name  # type: ignore[attr-defined]
    for _ in range(warmup):
        sess.run(None, {inp_name: X})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: X})
        ts.append(time.perf_counter() - t0)
    arr = np.asarray(ts, dtype=np.float64)
    return float(np.percentile(arr, 50) * 1000.0), float(np.percentile(arr, 95) * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/parity/features_oof.parquet")
    ap.add_argument("--onnx")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        print(f"[BENCH] onnxruntime not available: {e}")
        return 0

    feats_path = Path(args.features)
    if not feats_path.exists():
        print(f"[BENCH] features missing: {feats_path}")
        return 0
    X = np.ascontiguousarray(pd.read_parquet(feats_path).astype("float32").values)
    if X.size == 0:
        print("[BENCH] empty feature matrix; skipping")
        return 0

    # Resolve ONNX with broad search & debug
    onnx_path = args.onnx or os.environ.get("ONNX")
    candidates = []
    if onnx_path and os.path.exists(onnx_path):
        candidates.append(onnx_path)
    if os.path.exists("artifacts/models/latest.onnx"):
        candidates.append("artifacts/models/latest.onnx")
    import glob
    for p in sorted(glob.glob("reports/experiments/exp_*.json"), reverse=True):
        try:
            with open(p) as f:
                onnxp = (json.load(f).get("onnx") or {}).get("path")
            if onnxp:
                candidates.append(onnxp)
        except Exception:
            pass
    candidates += sorted(glob.glob("artifacts/models/**/*.onnx", recursive=True), reverse=True)
    for path in candidates:
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[BENCH][DEBUG] using ONNX: {path}")
            onnx_path = path
            break
    else:
        print(f"[BENCH] skip: no ONNX model found (checked {len(candidates)} candidates)")
        return 0
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore

    results: dict[str, dict[str, float]] = {}
    for b in (1, 32, 256):
        xb = X[: min(b, len(X))]
        p50_ms, p95_ms = _run(sess, xb)
        results[str(b)] = {"p50_ms": p50_ms, "p95_ms": p95_ms}

    out = {"onnx": onnx_path, "rows": int(X.shape[0]), "cols": int(X.shape[1]), "latency": results}
    print(json.dumps(out, indent=2))
    Path("reports/experiments").mkdir(parents=True, exist_ok=True)
    with open("reports/experiments/bench.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
