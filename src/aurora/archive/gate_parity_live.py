import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    side = json.loads(Path(args.sidecar).read_text())
    features = side.get("features") or side.get("feature_names")
    dtypes = side.get("dtypes") or ["float32"] * len(features)

    df = pd.read_csv(args.csv)
    # Expect exact feature columns in order
    if list(df.columns) != features:
        print(f"[GATE:PARITY][FAIL] schema mismatch: {list(df.columns)} vs {features}")
        return 1
    for c, dt in zip(df.columns, dtypes, strict=False):
        if str(df[c].dtype) != dt:
            df[c] = df[c].astype(dt)

    X = np.ascontiguousarray(df.astype("float32").values)

    # Native predictions (if available in CSV as a column 'pred_native'); otherwise skip
    if "pred_native" in df.columns:
        y_native = df["pred_native"].to_numpy(dtype=np.float32)
    else:
        print("[GATE:PARITY][WARN] pred_native column missing; skipping native vs onnx compare")
        return 0

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])  # type: ignore
    inp = sess.get_inputs()[0].name
    y_onnx = np.asarray(sess.run(None, {inp: X})[0]).reshape(-1).astype(np.float32)

    if y_onnx.shape != y_native.shape:
        print(f"[GATE:PARITY][FAIL] shape mismatch {y_native.shape} vs {y_onnx.shape}")
        return 1
    max_abs = float(np.max(np.abs(y_native - y_onnx)))
    p99 = float(np.percentile(np.abs(y_native - y_onnx), 99))
    print(json.dumps({"max_abs": max_abs, "p99_abs": p99}, indent=2))
    if max_abs > args.tol or p99 > args.tol:
        print(f"[GATE:PARITY][FAIL] parity exceeded tol={args.tol}")
        return 1
    print("PARITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


