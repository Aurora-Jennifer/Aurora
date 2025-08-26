import glob
import os

import numpy as np
import onnxruntime as ort
import pandas as pd


def latest(path_glob):
    g = sorted(glob.glob(path_glob))
    return g[-1] if g else None

def main():
    onnx_path = "artifacts/models/latest.onnx"
    if not os.path.exists(onnx_path):
        print("[DOCTOR] no latest.onnx")
        return 0
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    X = np.ascontiguousarray(pd.read_parquet("artifacts/parity/features_refit.parquet").astype("float32").values)
    y_nat = pd.read_parquet("artifacts/parity/preds_native_refit.parquet")["pred_native"].to_numpy().astype("float32")
    y_onnx = sess.run(None, {sess.get_inputs()[0].name: X})[0].reshape(-1).astype("float32")
    d = y_nat - y_onnx
    print(f"[DOCTOR] max|Δ|={np.max(np.abs(d)):.3e}  mean|Δ|={np.mean(np.abs(d)):.3e}")
    print(f"[DOCTOR] nat[:5]={y_nat[:5]}  onnx[:5]={y_onnx[:5]}  Δ[:5]={d[:5]}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
