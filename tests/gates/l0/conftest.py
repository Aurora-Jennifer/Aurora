"""
Session-scoped fixtures to load REAL artifacts produced by E2D.
If artifacts are missing, we (optionally) run a tiny E2D smoke to create them.

Env knobs:
- E2D_OUT: override path to E2D output dir (default: artifacts/run)
- E2D_CMD: override how to (re)generate outputs
- SKIP_E2D_RUN=1: never run E2D automatically; fail if outputs missing
"""

from __future__ import annotations
import os, subprocess
from pathlib import Path
import pandas as pd
import pytest

DEFAULT_E2D_OUT = Path(os.getenv("E2D_OUT", "artifacts/run"))
DEFAULT_E2D_CMD = os.getenv(
    "E2D_CMD",
    "FLAG_GOLDEN_SNAPSHOT_FROZEN=1 python -u scripts/e2d.py "
    "--profile config/profiles/golden_xgb_v2.yaml --out artifacts/run"
)

def _maybe_run_e2d(out_dir: Path) -> None:
    if os.getenv("SKIP_E2D_RUN", "0") == "1":
        return
    if not out_dir.exists() or not any(out_dir.glob("*.parquet")):
        # Best-effort generation; fail loud if command fails
        subprocess.check_call(DEFAULT_E2D_CMD, shell=True)

@pytest.fixture(scope="session")
def e2d_out_dir() -> Path:
    out = DEFAULT_E2D_OUT
    _maybe_run_e2d(out)
    assert out.exists(), f"E2D output dir missing: {out} (set SKIP_E2D_RUN=1 to skip auto-run)"
    return out

@pytest.fixture(scope="session")
def features_path(e2d_out_dir: Path) -> Path:
    # Use golden snapshot features for L0 gate testing
    golden_features = Path("artifacts/snapshots/golden_ml_v1/features.parquet")
    if golden_features.exists():
        return golden_features
    
    # Fallback to E2D output if golden features don't exist
    candidates = [
        e2d_out_dir / "features.parquet",
        e2d_out_dir / "X.parquet",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        found = list(e2d_out_dir.glob("*.parquet"))
        assert found, f"No parquet features found in {e2d_out_dir} or golden snapshot"
        p = found[0]
    return p

@pytest.fixture(scope="session")
def features_df(features_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    # If timestamp is a column, make it the index for time contracts
    for col in ("timestamp", "ts", "datetime"):
        if col in df.columns:
            df = df.set_index(col)
            break
    return df

@pytest.fixture(scope="session")
def snapshot_dir() -> Path:
    p = Path("artifacts/snapshots/golden_ml_v1")
    assert p.exists(), f"Golden snapshot not found at {p}"
    return p
