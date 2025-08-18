import json
from pathlib import Path
import numpy as np
import pickle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ml.models.dummy_model import DummyModel  # ensure pickle resolves


def test_model_dummy_v1_golden():
    art = Path("artifacts/models/dummy_v1.pkl")
    if not art.exists():
        # allow skip if artifact not present
        return
    model = pickle.loads(art.read_bytes())
    base = json.loads(Path("baselines/model_dummy_v1.json").read_text())
    X = np.array(base["features"], dtype="float64")
    scores = [float(model.predict(X[i:i+1])[0]) for i in range(X.shape[0])]
    for a, b in zip(scores, base["scores"]):
        assert abs(a - float(b)) <= 1e-9


