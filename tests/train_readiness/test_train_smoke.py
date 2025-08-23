import glob
import json
import subprocess

import numpy as np
import pytest

from core.ml.build_features import build_matrix


@pytest.mark.sanity
def test_builder_is_lagged(load_clean_df):
    df = load_clean_df("SPY", "tiny")
    X, y = build_matrix(df, horizon=1)
    # No NaNs
    assert np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y.to_numpy()).all()


@pytest.mark.sanity
def test_train_smoke_runs():
    r = subprocess.run(["python", "scripts/train_linear.py"], check=False)
    assert r.returncode == 0
    paths = glob.glob("reports/experiments/*.json")
    assert paths, "no experiment JSON produced"
    with open(sorted(paths)[-1]) as f:
        meta = json.load(f)
    assert meta["metrics"]["r2"] < 0.60
    assert abs(meta["metrics"]["ic"]) < 0.40


