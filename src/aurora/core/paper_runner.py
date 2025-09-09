from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import contextlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

from brokers.paper import PaperBroker
from ml.model_interface import ModelSpec
from ml.registry import load_model
from ml.runtime import (
    build_features,
    compute_turnover,
    detect_weight_spikes,
    infer_weights,
    set_seeds,
)
from utils.ops_runtime import kill_switch, notify_ntfy

try:
    from tools.provenance import write_provenance
except Exception:
    import os
    import sys

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

    print("🚀 Starting paper trading simulation")
    print(f"📈 Symbols: {args.symbols}")
    print(f"⏰ Poll interval: {args.poll_sec} seconds")
    print(f"💰 Starting cash: ${args.cash:,.2f}")
    print(f"⚙️  Profile: {args.profile}")
    print()

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
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

    print("✅ Initialized paper broker and metadata")
    print()
    
    # Reality check banner (if enabled)
    try:
        cfg = yaml.safe_load(Path("config/base.yaml").read_text())
        if cfg.get("flags", {}).get("show_reality_check_on_start", False):
            print("=== REALITY CHECK ===")
            print("To be actually valuable, add:")
            print("• Realtime data • OMS • Risk v2 • Proper BT • Better features/models")
            print("See docs/REALITY_CHECK.md")
            print()
    except Exception:
        pass

    # Optional model runtime (feature-flagged)
    STATE = Path("reports/runner_state.json")

    def _load_prev_weights() -> dict:
        try:
            if STATE.exists():
                return json.loads(STATE.read_text()).get("prev_weights", {})
        except Exception:
            pass
        return {}

    def _save_prev_weights(prev_weights: dict) -> None:
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({"prev_weights": prev_weights}, indent=2))
        except Exception:
            pass

    try:
        print("🔍 Loading configuration...")
        cfg = yaml.safe_load(Path("config/base.yaml").read_text())
        models_cfg = (cfg or {}).get("models", {}) or {}
        
        if models_cfg.get("enable", False):
            print("🤖 Model system enabled - loading models...")
            set_seeds(1337)
            reg = yaml.safe_load(Path("config/models.yaml").read_text())["registry"]
            m_id = models_cfg["selected"]
            spec_cfg = reg[m_id]
            spec = ModelSpec(
                kind=spec_cfg["kind"], path=spec_cfg["path"], metadata=spec_cfg.get("metadata", {})
            )
            model, art_sha = load_model(spec)
            print(f"✅ Loaded model: {m_id} ({spec.kind})")
            
            feats_list = models_cfg.get("input_features", [])
            feat_order = (spec.metadata or {}).get("feature_order", feats_list)
            min_bars = int(models_cfg.get("min_history_bars", 120))
            print(f"📊 Features: {len(feats_list)} features, {min_bars} min bars")
            
            # Load cached CI data if present; fallback to synthetic drift
            weights_by_symbol: dict[str, float] = {}
            prev_weights: dict[str, float] | None = _load_prev_weights()
            model_fallbacks = 0
            
            print("📈 Processing symbols...")
            for sym in meta["symbols"]:
                print(f"  🔍 Processing {sym}...")
                cache_pq = Path("data/smoke_cache") / f"{sym}.parquet"
                if cache_pq.exists():
                    df = pd.read_parquet(cache_pq)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    print(f"    ✅ Using cached data: {len(df)} bars")
                else:
                    # synthetic small series
                    idx = pd.date_range("2020-01-01", periods=180, tz="UTC")
                    # deterministic slight trend
                    close = pd.Series(100.0, index=idx)
                    close = close * (1.0 + 0.0005) ** np.arange(len(idx))
                    df = pd.DataFrame({"Close": close.values}, index=idx)
                    print(f"    ⚠️  Using synthetic data: {len(df)} bars")
                    
                if "Close" not in df.columns and df.shape[1] > 0:
                    # pick first as close-like
                    df = pd.DataFrame({"Close": df.iloc[:, 0]}, index=df.index)
                    
                F = build_features(df, feats_list)
                print(f"    📊 Built {F.shape[1]} features")
                try:
                    print(f"    🧮 Feature rows after dropna: {len(F)}")
                except Exception:
                    pass
                # Diagnostics: show expected vs built for alignment
                try:
                    built_cols = list(F.columns)
                    print(f"    🔎 Expected order: {feat_order}")
                    print(f"    🔎 Built columns: {built_cols}")
                except Exception:
                    pass
                
                w = infer_weights(
                    model,
                    F,
                    feat_order,
                    models_cfg.get("score_to_weight", "tanh"),
                    float(models_cfg.get("max_abs_weight", 0.5)),
                    min_bars,
                )
                
                # Apply risk v2 layer if enabled
                if isinstance(w, dict) and "status" not in w:
                    from ml.runtime import apply_risk_layer
                    
                    # Create decision context for risk layer
                    decision = {"symbol": sym, "weight": float(next(iter(w.values())))}
                    context = {
                        "bars": {sym: df},
                        "account": {"equity": args.cash},
                        "positions": prev_weights or {},
                        "portfolio": {"gross_weight": sum(abs(w) for w in (prev_weights or {}).values())}
                    }
                    
                    decision = apply_risk_layer(decision, context, cfg)
                    w = {0: decision["weight"]}
                    
                    # Log risk telemetry
                    if "risk" in decision:
                        risk_info = decision["risk"]
                        print(f"    🛡️  Risk: ATR={risk_info.get('atr', 'N/A'):.4f}, "
                              f"Stop={risk_info.get('stop', 'N/A'):.2f}, "
                              f"Action={risk_info.get('action', 'N/A')}")
                        if risk_info.get("veto"):
                            print(f"    ⚠️  Risk veto: {risk_info.get('reason', 'unknown')}")
                # map index 0 weight to symbol for now
                if isinstance(w, dict) and "status" not in w:
                    # use first weight
                    weight = float(next(iter(w.values())))
                    weights_by_symbol[sym] = weight
                    print(f"    🎯 Weight: {weight:.3f}")
                else:
                    model_fallbacks += 1
                    # Print detailed fallback reason if available
                    if isinstance(w, dict) and "status" in w:
                        print(f"    ⚠️  Fallback ({w.get('status')}): {w.get('reason')}")
                    else:
                        print("    ⚠️  Fallback (no valid weight)")
                    
            print(f"📊 Total weights: {len(weights_by_symbol)} symbols, {model_fallbacks} fallbacks")
            
            # Tripwires (only evaluate if we have a previous snapshot)
            MAX_DW = 0.25
            spikes = detect_weight_spikes(prev_weights, weights_by_symbol, MAX_DW)
            if spikes:
                meta["model_tripwire"] = {"reason": "weight_spike", "spikes": spikes}
                model_fallbacks += 1
                print(f"🚨 Weight spike detected: {spikes}")
                
            TURNOVER_CAP = 1.0
            if prev_weights:
                turnover = compute_turnover(prev_weights, weights_by_symbol)
                if turnover > TURNOVER_CAP:
                    meta["model_tripwire_turnover"] = {"turnover": turnover}
                    model_fallbacks += 1
                    print(f"🔄 High turnover detected: {turnover:.3f}")
                    
            meta.update(
                {
                    "model_id": m_id,
                    "model_kind": spec.kind,
                    "artifact_sha256": art_sha,
                    "feature_order": feat_order,
                    "model_enabled": True,
                    "model_fallbacks": model_fallbacks,
                }
            )
            # weights_by_symbol is ready for router integration (future step)
            prev_weights = weights_by_symbol.copy()
            print("✅ Model processing complete")
        else:
            print("⚠️  Model system disabled - running in fallback mode")
            
    except Exception as e:
        # Non-fatal: log via meta note and continue fallback
        meta["model_runtime"] = "fallback"
        print(f"⚠️  Model loading failed: {e}")
        print("   Continuing in fallback mode...")

    print()
    print("⏰ Starting main loop...")
    print("   (Press Ctrl+C to stop)")
    print()

    for stopped in kill_switch(interval_s=args.poll_sec):
        if stopped:
            print("🛑 Kill switch triggered")
            break
        # integrate signal + routing here in next iteration

    print("✅ Main loop completed")

    meta["stop"] = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    # Persist prev weights across restarts
    with contextlib.suppress(Exception):
        _save_prev_weights(locals().get("prev_weights") or {})
        
    print("💾 Results saved to reports/paper_run.meta.json")
    
    # End-of-run notification with summary if anomalies occurred
    if args.ntfy:
        fallbacks = int(meta.get("model_fallbacks", 0))
        abnormal = fallbacks > 0 or ("model_tripwire" in meta or "model_tripwire_turnover" in meta)
        title = "Aurora: paper OK" if not abnormal else "Aurora: paper WARN"
        body = {
            "run_id": run_id,
            "fallbacks": fallbacks,
            "tripwire": meta.get("model_tripwire"),
            "turnover": meta.get("model_tripwire_turnover"),
        }
        notify_ntfy(title, body)
        # Auto-issue on anomalies if configured
        if abnormal:
            repo = os.getenv("GITHUB_REPOSITORY", "")
            token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
            if shutil.which("python") and (repo and token):
                with contextlib.suppress(Exception):
                    subprocess.run(["python", "tools/gh_issue.py", repo, token], check=False)


if __name__ == "__main__":
    main()
