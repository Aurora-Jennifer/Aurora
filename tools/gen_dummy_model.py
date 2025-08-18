import pickle
from pathlib import Path
import numpy as np


class DummyModel:
    def __init__(self, w):
        self.w = np.array(w, dtype="float64")

    def predict(self, X):
        X = np.asarray(X, dtype="float64")
        # simple linear projection on last row
        return X[-1].dot(self.w) * np.ones(1)


if __name__ == "__main__":
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    with open("artifacts/models/dummy_v1.pkl", "wb") as f:
        pickle.dump(DummyModel([0.5, -0.2, 0.1]), f)
    print("wrote artifacts/models/dummy_v1.pkl")


