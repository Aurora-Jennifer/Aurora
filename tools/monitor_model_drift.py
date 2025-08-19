from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ml.feature_stats import psi, stats  # noqa: E402


def _deterministic_prices(n: int = 360, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(1337)
    rets = rng.normal(0.0002, 0.005, size=n)
    close = start * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"Close": close}, index=idx)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats: dict[str, Any] = {}
    feats["ret_1d"] = df["Close"].pct_change()
    feats["ret_5d"] = df["Close"].pct_change(5)
    feats["vol_10d"] = df["Close"].pct_change().rolling(10).std()
    return pd.DataFrame(feats, index=df.index).dropna()


def main() -> None:
    cfg = yaml.safe_load(Path("config/base.yaml").read_text())
    reg = yaml.safe_load(Path("config/models.yaml").read_text())["registry"]
    models_cfg = cfg.get("models", {}) or {}
    selected = models_cfg.get("selected", "dummy_v1")
    model_cfg = reg[selected]

    training_stats = model_cfg.get("training_stats") or {}
    thresholds = {
        "psi_warn": model_cfg.get("psi_warn", 0.10),
        "psi_fail": model_cfg.get("psi_fail", 0.25),
        "max_missing_pct": model_cfg.get("max_missing_pct", 0.01),
    }

    prices = _deterministic_prices()
    feat_df = _build_features(prices)
    cur_stats = stats(feat_df)
    cur_psi = psi(feat_df, training_stats)

    out = {
        "model": {
            "id": selected,
            "artifact_sha256": model_cfg.get("artifact_sha256", ""),
        },
        "stats": cur_stats,
        "psi": {k: v for k, v in cur_psi.items() if k != "__global__"},
        "psi_global": cur_psi.get("__global__", float("nan")),
        "thresholds": thresholds,
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports") / "model_drift.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
