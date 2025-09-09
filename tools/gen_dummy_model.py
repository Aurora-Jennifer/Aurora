import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ml.models.dummy_model import DummyModel

if __name__ == "__main__":
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    with open("artifacts/models/dummy_v1.pkl", "wb") as f:
        pickle.dump(DummyModel([0.5, -0.2, 0.1]), f)
    print("wrote artifacts/models/dummy_v1.pkl")
