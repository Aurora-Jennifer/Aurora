import numpy as np
import pandas as pd

PRICE_COLS = ("Open", "High", "Low", "Close")

def _z(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std(ddof=0)
    return (x - mu) / (sd + 1e-12)

def additive_invariance_score(df_base: pd.DataFrame, df_shift: pd.DataFrame) -> float:
    # 1) align by index (intersection, sorted)
    common = df_base.index.intersection(df_shift.index)
    if len(common) < 3:
        return 0.0
    b = df_base.loc[common].sort_index()
    s = df_shift.loc[common].sort_index()

    # 2) compute per-column z-scored first differences and average cosine similarity
    scores = []
    for col in PRICE_COLS:
        if col in b.columns and col in s.columns:
            dy = np.diff(b[col].to_numpy(dtype=np.float64))
            dz = np.diff(s[col].to_numpy(dtype=np.float64))
            if dy.size < 2:  # not enough points to be meaningful
                continue
            y = _z(dy)
            z = _z(dz)
            scores.append(float(np.clip(np.dot(y, z) / (len(y)), -1.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0

def invariance_score(y_hat, y_hat_t):
    y = np.asarray(y_hat, dtype=np.float64)
    z = np.asarray(y_hat_t, dtype=np.float64)

    # z-score both (additive + scale invariance), numerically stable
    y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    z = (z - z.mean()) / (z.std(ddof=0) + 1e-12)

    # cosine similarity == correlation after z-scoring
    return float(np.clip((y * z).mean(), -1.0, 1.0))
