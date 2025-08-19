from __future__ import annotations

import argparse
import json
import os
import os as _os
import sys as _sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from core.execution.canary_limits import CanaryConfig, check_caps
from ml.model_interface import ModelSpec
from ml.registry import load_model
from ml.runtime import build_features, infer_weights, set_seeds
from utils.ops_runtime import kill_switch, notify_ntfy
from utils.quotes_provider import QuoteProvider, get_quote_provider

STATE = Path("reports/runner_state.json")


def _load_prev_weights():
    try:
        return json.loads(STATE.read_text()).get("prev_weights", {})
    except Exception:
        return {}


def _save_prev_weights(prev):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"prev_weights": prev}, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,TSLA")
    ap.add_argument("--poll-sec", type=float, default=5.0)
    ap.add_argument("--profile", default="live_canary")
    ap.add_argument("--shadow", action="store_true", default=True)
    ap.add_argument("--ntfy", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="Max loop iterations (0=unbounded)")
    ap.add_argument(
        "--quotes", type=str, default="dummy", choices=["dummy", "ibkr"], help="Quote provider type"
    )
    args = ap.parse_args(argv)

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    Path("logs/canary").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/canary") / (datetime.now(UTC).strftime("%Y-%m-%d") + ".jsonl")

    def _deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in (b or {}).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    cfg = yaml.safe_load(Path("config/base.yaml").read_text())
    # Optional overlay profile (YAML): config/profiles/<profile>.yaml
    overlay_path = Path("config/profiles") / f"{args.profile}.yaml"
    if overlay_path.exists():
        overlay_cfg = yaml.safe_load(overlay_path.read_text())
        cfg = _deep_merge(cfg, overlay_cfg)
    models_cfg = cfg.get("models") or {}
    live = cfg.get("live") or {}
    paper = cfg.get("paper") or {}
    cfg.get("risk") or {}

    equity = float((cfg.get("live") or {}).get("equity", 100_000.0))
    cap_cfg = CanaryConfig(
        equity=equity,
        per_trade_notional_pct=float(live.get("per_trade_notional_pct", 0.01)),
        notional_daily_cap_pct=float(live.get("notional_daily_cap_pct", 0.10)),
    )
    spike_cap = float(paper.get("weight_spike_cap", 0.2))
    turnover_cap = float(paper.get("turnover_cap", 0.8))

    prev_weights = _load_prev_weights()
    qp: QuoteProvider = get_quote_provider(args.quotes)

    # Quote heartbeat tracking
    quote_miss_count = 0

    set_seeds(1337)
    reg = yaml.safe_load(Path("config/models.yaml").read_text())["registry"]
    m_id = models_cfg.get("selected", "dummy_v1")
    spec_cfg = reg[m_id]
    spec = ModelSpec(
        kind=spec_cfg["kind"], path=spec_cfg["path"], metadata=spec_cfg.get("metadata", {})
    )
    model, art_sha = load_model(spec)
    feats_list = models_cfg.get("input_features", [])
    feat_order = (spec.metadata or {}).get("feature_order", feats_list)
    min_bars = int(models_cfg.get("min_history_bars", 60))

    def build_prices(symbol: str) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=200, tz="UTC")
        close = pd.Series(100.0, index=idx) * (1.0 + 0.0005) ** np.arange(len(idx))
        return pd.DataFrame({"Close": close.values}, index=idx)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    weights_now: dict[str, float] = {}
    fallbacks = 0
    anomalies: list[str] = []
    step = 0

    for stopped in kill_switch(interval_s=args.poll_sec):
        if stopped:
            break
        step += 1
        if args.steps and step >= args.steps:
            break

        # Check for session lockout
        if Path("runlocks/live.lock").exists():
            print("Session locked due to previous anomaly/fallback")
            break
        for sym in symbols:
            F = build_features(build_prices(sym), feats_list)
            w = infer_weights(
                model,
                F,
                feat_order,
                models_cfg.get("score_to_weight", "tanh"),
                float(models_cfg.get("max_abs_weight", 0.5)),
                min_bars,
            )
            if isinstance(w, dict) and "status" not in w:
                weights_now[sym] = float(next(iter(w.values())))
            else:
                fallbacks += 1

        def _turnover(a: dict[str, float], b: dict[str, float]) -> float:
            keys = set(a.keys()) | set(b.keys())
            return sum(abs(float(b.get(k, 0.0)) - float(a.get(k, 0.0))) for k in keys)

        # Pre-trade tripwires
        for sym in symbols:
            try:
                quote = qp.quote(sym)
                now_ts = datetime.now(UTC)
                quote_ts = datetime.fromisoformat(quote["ts"].replace("Z", "+00:00"))
                age_ms = int((now_ts - quote_ts).total_seconds() * 1000)

                # Stale quote tripwire
                if age_ms > 1500:
                    anomalies.append(f"stale_quote:{sym}:{age_ms}ms")
                    continue

                # Wide spread tripwire
                bid, ask, mid = quote["bid"], quote["ask"], quote["mid"]
                if bid and ask and mid and bid == bid and ask == ask and mid == mid:
                    spread_bps = ((ask - bid) / mid) * 10000
                    if spread_bps > 50:  # 50 bps = 0.5%
                        anomalies.append(f"wide_spread:{sym}:{spread_bps:.1f}bps")
                        continue

                # Market state tripwire (null/zero quotes)
                if not bid or not ask or bid == 0 or ask == 0:
                    anomalies.append(f"market_closed:{sym}")
                    continue

            except Exception as e:
                anomalies.append(f"quote_error:{sym}:{str(e)[:50]}")

        # Quote heartbeat check
        import math

        all_quotes_valid = True
        for sym in symbols:
            try:
                quote = qp.quote(sym)
                if not math.isfinite(quote.get("mid", float("nan"))):
                    all_quotes_valid = False
                    break
            except Exception:
                all_quotes_valid = False
                break

        if not all_quotes_valid:
            quote_miss_count += 1
        else:
            quote_miss_count = 0

        if quote_miss_count >= 5:
            anomalies.append("quote_heartbeat_lost")

        spikes = [
            s
            for s in symbols
            if abs(weights_now.get(s, 0.0) - float(prev_weights.get(s, 0.0))) > spike_cap
        ]
        t_over = (
            _turnover(prev_weights, weights_now)
            if prev_weights
            else sum(abs(v) for v in weights_now.values())
        )
        if spikes:
            anomalies.append(f"weight_spike:{','.join(spikes)}")
        if t_over > turnover_cap:
            anomalies.append(f"turnover:{t_over:.3f}")

        # Session circuit breaker (multi-anomaly lockout)
        if anomalies or fallbacks > 0:
            notify_ntfy("Aurora live canary: anomaly/fallback → session lockout")
            # Write lock file for session termination
            Path("runlocks").mkdir(exist_ok=True)
            Path("runlocks/live.lock").write_text("anomaly")
            break

        day_used = 0.0
        with log_path.open("a") as lf:
            for sym in symbols:
                mid = float(qp.quote(sym)["mid"])
                target_w = float(weights_now.get(sym, 0.0))
                prev_w = float(prev_weights.get(sym, 0.0))
                delta_w = target_w - prev_w
                side = "BUY" if delta_w > 0 else ("SELL" if delta_w < 0 else "HOLD")
                qty = abs(delta_w) * equity / max(mid, 1e-9)
                decision = check_caps(sym, qty, mid, day_used, cap_cfg)
                reasons = []
                if decision.action != "OK":
                    reasons.append(decision.reason or "cap")
                    anomalies.append(decision.reason or "cap")
                line = {
                    "ts": datetime.now(UTC).isoformat(),
                    "symbol": sym,
                    "side": side,
                    "qty": float(qty),
                    "w_prev": prev_w,
                    "w_now": target_w,
                    "px_theory": mid,
                    "px_quote_mid": mid,
                    "notional": float(qty * mid),
                    "reasons": reasons,
                }
                lf.write(json.dumps(line) + "\n")
                day_used += abs(qty * mid)
        prev_weights = weights_now.copy()
        # break  # single-iteration smoke; normal runs rely on kill-switch

    _save_prev_weights(prev_weights)
    meta = {
        "run_id": run_id,
        "profile": args.profile,
        "symbols": symbols,
        "model_enabled": bool(models_cfg.get("enable", False)),
        "model": {"id": m_id, "artifact_sha256": art_sha},
        "model_fallbacks": fallbacks,
        "anomalies": anomalies,
    }
    Path("reports/canary_run.meta.json").write_text(json.dumps(meta, indent=2))
    if args.ntfy and (fallbacks > 0 or anomalies):
        notify_ntfy(
            "Aurora: canary WARN",
            {"run_id": run_id, "fallbacks": fallbacks, "anomalies": anomalies},
        )
    # Auto-issue on anomalies if configured
    if fallbacks > 0 or anomalies:
        repo = os.getenv("GITHUB_REPOSITORY", "")
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
        if shutil.which("python") and (repo and token):
            try:
                subprocess.run(["python", "tools/gh_issue.py", repo, token], check=False)
            except Exception:
                pass
    if fallbacks > 0 or anomalies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
