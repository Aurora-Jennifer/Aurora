from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from brokers.paper import PaperBroker
from utils.ops_runtime import kill_switch, notify_ntfy
import yaml
import pandas as pd
from ml.model_interface import ModelSpec
from ml.registry import load_model
from ml.runtime import set_seeds, build_features, infer_weights

try:
    from tools.provenance import write_provenance
except Exception:
    import os, sys
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    try:
        from provenance import write_provenance  # type: ignore
    except Exception:
        def write_provenance(*_a, **_k):
            return {}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,TSLA")
    ap.add_argument("--poll-sec", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--profile", default="risk_balanced")
    ap.add_argument("--ntfy", action="store_true")
    args = ap.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    Path("reports").mkdir(parents=True, exist_ok=True)
    broker = PaperBroker()
    meta = {
        "run_id": run_id,
        "profile": args.profile,
        "symbols": args.symbols.split(","),
        "start": run_id,
    }
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    write_provenance("reports/paper_provenance.json", ["config/base.yaml"])

    # Optional model runtime (feature-flagged)
    try:
        cfg = yaml.safe_load(Path("config/base.yaml").read_text())
        models_cfg = (cfg or {}).get("models", {}) or {}
        if models_cfg.get("enable", False):
            set_seeds(1337)
            reg = yaml.safe_load(Path("config/models.yaml").read_text())["registry"]
            m_id = models_cfg["selected"]
            spec_cfg = reg[m_id]
            spec = ModelSpec(kind=spec_cfg["kind"], path=spec_cfg["path"], metadata=spec_cfg.get("metadata", {}))
            model, art_sha = load_model(spec)
            feats_list = models_cfg.get("input_features", [])
            feat_order = (spec.metadata or {}).get("feature_order", feats_list)
            min_bars = int(models_cfg.get("min_history_bars", 120))
            # Load cached CI data if present; fallback to synthetic drift
            weights_by_symbol = {}
            for sym in meta["symbols"]:
                cache_pq = Path("data/smoke_cache") / f"{sym}.parquet"
                if cache_pq.exists():
                    df = pd.read_parquet(cache_pq)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                else:
                    # synthetic small series
                    idx = pd.date_range("2020-01-01", periods=180, tz="UTC")
                    close = pd.Series(100.0, index=idx).cumprod() * 0 + 100.0
                    df = pd.DataFrame({"Close": close.values}, index=idx)
                if "Close" not in df.columns and df.shape[1] > 0:
                    # pick first as close-like
                    df = pd.DataFrame({"Close": df.iloc[:, 0]}, index=df.index)
                F = build_features(df, feats_list)
                w = infer_weights(model, F, feat_order, models_cfg.get("score_to_weight", "tanh"), float(models_cfg.get("max_abs_weight", 0.5)), min_bars)
                # map index 0 weight to symbol for now
                if isinstance(w, dict) and "status" not in w:
                    # use first weight
                    weights_by_symbol[sym] = float(next(iter(w.values())))
            meta.update({
                "model_id": m_id,
                "model_kind": spec.kind,
                "artifact_sha256": art_sha,
                "feature_order": feat_order,
            })
            # weights_by_symbol is ready for router integration (future step)
    except Exception:
        # Non-fatal: log via meta note and continue fallback
        meta["model_runtime"] = "fallback"

    for stopped in kill_switch(interval_s=args.poll_sec):
        if stopped:
            break
        # integrate signal + routing here in next iteration

    meta["stop"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    if args.ntfy:
        notify_ntfy("Aurora: paper done", {"run_id": run_id})


if __name__ == "__main__":
    main()


