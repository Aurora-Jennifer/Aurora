import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    art = Path("artifacts/models/dummy_v1.pkl")
    if not art.exists():
        raise SystemExit("dummy model not found; run tools/gen_dummy_model.py first")
    model = pickle.loads(art.read_bytes())
    # Fixed 3x3 feature matrix aligned with [ret_1d, ret_5d, vol_10d]
    X = np.array(
        [
            [0.001, 0.005, 0.010],
            [0.002, -0.004, 0.015],
            [-0.001, 0.003, 0.008],
        ],
        dtype="float64",
    )
    # Predict per row for stability
    scores = [float(model.predict(X[i : i + 1])[0]) for i in range(X.shape[0])]
    out = {
        "features": X.tolist(),
        "scores": scores,
        "feature_order": ["ret_1d", "ret_5d", "vol_10d"],
    }
    Path("baselines").mkdir(parents=True, exist_ok=True)
    Path("baselines/model_dummy_v1.json").write_text(json.dumps(out, indent=2))
    print("Blessed baselines/model_dummy_v1.json")


if __name__ == "__main__":
    main()
