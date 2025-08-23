#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def assert_schema(expected, df: pd.DataFrame):
    cols = list(df.columns)
    if cols != expected:
        raise SystemExit(f"[PARITY] schema mismatch.\nexpected={expected}\nactual  ={cols}")


def run_onnx(onnx_path: str, X: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except Exception as e:
        raise SystemExit(f"[PARITY] onnxruntime not available: {e}") from e
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore
    inp = {sess.get_inputs()[0].name: X}
    out = sess.run(None, inp)[0]
    return np.asarray(out).reshape(-1)


def _resolve_onnx_path(cli: str | None) -> str | None:
    if cli and os.path.exists(cli):
        return cli
    candidates = []
    if os.path.exists("artifacts/models/latest.onnx"):
        candidates.append("artifacts/models/latest.onnx")
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
            print(f"[PARITY][DEBUG] using ONNX: {path}")
            return path
    print(f"[PARITY] skip: no ONNX model found (checked {len(candidates)} candidates)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--sidecar", default="artifacts/parity/sidecar.json")
    ap.add_argument("--native_preds")
    ap.add_argument("--onnx")
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    # Defaults: prefer refit artifacts
    features_path = args.features or (
        "artifacts/parity/features_refit.parquet" if os.path.exists("artifacts/parity/features_refit.parquet")
        else "artifacts/parity/features_oof.parquet"
    )
    native_path = args.native_preds or (
        "artifacts/parity/preds_native_refit.parquet" if os.path.exists("artifacts/parity/preds_native_refit.parquet")
        else "artifacts/parity/preds_native.parquet"
    )
    onnx_path = _resolve_onnx_path(args.onnx)
    if not onnx_path:
        print("[PARITY] skip: no ONNX model found")
        return 0

    with open(args.sidecar) as f:
        side = json.load(f)
    expected = side.get("features") or side.get("feature_names")
    if not expected:
        raise SystemExit("[PARITY] sidecar missing 'features'")

    feat_path = Path(features_path)
    if not feat_path.exists():
        raise SystemExit(f"[PARITY] features file missing: {feat_path}")
    df = pd.read_parquet(feat_path) if feat_path.suffix == ".parquet" else pd.read_csv(feat_path)

    assert_schema(expected, df)
    # Robust contiguous float32 array across pandas versions
    X = np.ascontiguousarray(df.astype("float32").values)

    y_onnx = run_onnx(onnx_path, X)

    pred_path = Path(native_path)
    if not pred_path.exists():
        raise SystemExit(f"[PARITY] native preds missing: {pred_path}")
    if pred_path.suffix == ".parquet":
        y_nat = pd.read_parquet(pred_path).iloc[:, -1].to_numpy()
    else:
        y_nat = pd.read_csv(pred_path).iloc[:, -1].to_numpy()

    y_nat = y_nat.astype("float32", copy=False)
    y_onnx = y_onnx.astype("float32", copy=False)

    if y_nat.shape != y_onnx.shape:
        raise SystemExit(f"[PARITY] shape mismatch {y_nat.shape} vs {y_onnx.shape}")

    diff = float(np.max(np.abs(y_nat - y_onnx))) if y_nat.size else 0.0
    ok = np.allclose(y_nat, y_onnx, atol=args.atol, rtol=0.0)
    if not ok:
        raise SystemExit(f"[PARITY] numeric fail: max|Δ|={diff:.3e} atol={args.atol}")

    # Emit parity status for CI/review
    os.makedirs("reports/experiments", exist_ok=True)
    with open("reports/experiments/parity.json", "w") as f:
        json.dump({
            "ok": bool(ok),
            "max_abs_diff": float(np.max(np.abs(y_nat - y_onnx))),
            "nrows": int(len(y_nat))
        }, f, indent=2)
    print(f"[PARITY] {'OK' if ok else 'FAIL'} numeric: max|Δ|={diff:.3e} on {len(y_nat)} rows")
    return 0 if ok else 1  # keep script's exit meaningful; Makefile already allow-fails


if __name__ == "__main__":
    raise SystemExit(main())


