import hashlib

import numpy as np


def hash_array(X: np.ndarray) -> str:
    """Compute a robust hash for a numpy array (shape, dtype, subsampled bytes)."""
    h = hashlib.sha256()
    h.update(str((X.shape, str(X.dtype))).encode())
    if X.size == 0:
        return h.hexdigest()
    rows = np.linspace(0, X.shape[0] - 1, num=min(1024, X.shape[0]), dtype=int)
    cols = np.arange(min(128, X.shape[1])) if X.ndim > 1 else None
    sample = X[np.ix_(rows, cols)] if cols is not None else X[rows]
    h.update(np.ascontiguousarray(sample).tobytes())
    return h.hexdigest()


