import numpy as np


class DummyModel:
    def __init__(self, w):
        self.w = np.array(w, dtype="float64")

    def predict(self, X):
        X = np.asarray(X, dtype="float64")
        # simple linear projection on last row
        return X[-1].dot(self.w) * np.ones(1)
