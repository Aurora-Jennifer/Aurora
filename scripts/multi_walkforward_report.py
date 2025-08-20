#!/usr/bin/env python3
"""
# © 2025 Aurora Analytics — Proprietary
# No redistribution, public posting, or model-training use.
# Internal-only. Ref: AUR-NOTICE-2025-01

Multi-profile, multi-symbol walkforward runner that uses the existing
walkforward framework APIs and writes a markdown report.
"""

import argparse
import importlib
import contextlib
import datetime as dt
import json
import logging
import math
import os
import pathlib
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from tools.provenance import write_provenance
except Exception:  # fallback when package import fails
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    try:
        from provenance import write_provenance  # type: ignore
    except Exception:

        def write_provenance(*_a, **_k):
            return {}


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.walkforward_framework import (
    LeakageProofPipeline,
    build_feature_table,
    gen_walkforward,
    walkforward_run,
)

# Data sanity (fast pre-check for smoke)
try:
    DataSanityValidator = importlib.import_module("core.data_sanity").DataSanityValidator  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    DataSanityValidator = None  # type: ignore


def in_ci() -> bool:
    return str(os.getenv("CI", "")).lower() in {"1", "true", "yes", "on"}


def _write_smoke_json(payload: dict) -> None:
    payload = {
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        **payload,
    }
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_run.json").write_text(json.dumps(payload, indent=2))


def _git_sha_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_smoke_meta(payload: dict) -> None:
    meta = {
        "run_id": payload.get("timestamp"),
        "git_sha": _git_sha_short(),
        "profile": "walkforward_ci" if in_ci() else "walkforward",
        "symbols": payload.get("symbols"),
        "folds": payload.get("folds"),
        "seed": 1337,
    }
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_run.meta.json").write_text(json.dumps(meta, indent=2))


def _write_run_json(payload: dict) -> None:
    # Build a minimal run.json for promotion validation
    started = payload.get("started_at") or dt.datetime.now(dt.UTC).isoformat()
    finished = payload.get("finished_at") or dt.datetime.now(dt.UTC).isoformat()
    cfg_hash = ""
    try:
        base = pathlib.Path("config/base.yaml")
        if base.exists():
            import hashlib

            h = hashlib.sha256()
            h.update(base.read_bytes())
            cfg_hash = h.hexdigest()
    except Exception:
        cfg_hash = ""
    run = {
        "run_id": payload.get("timestamp") or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        "model_version": "n/a",
        "data_hash": payload.get("data_hash", ""),
        "config_hash": cfg_hash,
        "started_at": started,
        "finished_at": finished,
    }
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run, indent=2))


def _map_violation(result) -> tuple[str, str]:
    if not getattr(result, "violations", None):
        return ("UNKNOWN", "data_sanity_violation")
    v0 = result.violations[0]
    code = getattr(v0, "code", "UNKNOWN")
    details = getattr(v0, "details", getattr(v0, "msg", str(v0)))
    return (code, f"data_sanity_violation[{code}]: {details}")


def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Check for offline mode (tests or CI)
    offline_mode = os.getenv("AURORA_TEST_OFFLINE", "1") == "1" or os.getenv("CI") == "true"
    
    if offline_mode:
        # Try fixture directory first
        fixture_path = pathlib.Path("data/fixtures/smoke") / f"{symbol}.csv"
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            # Ensure OHLCV columns exist
            cols = set(df.columns)
            if "Open" in cols and "High" in cols and "Low" in cols and "Close" in cols and "Volume" in cols:
                return df[["Open", "High", "Low", "Close", "Volume"]]
        
        # Fall back to CI cache
        cache_path = pathlib.Path("data/smoke_cache") / f"{symbol}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            # If only Close exists, synthesize minimal OHLCV
            cols = set(df.columns)
            if (
                "Open" not in cols
                or "High" not in cols
                or "Low" not in cols
                or "Volume" not in cols
            ):
                close = (
                    df[df.columns[0]] if len(df.columns) == 1 else df.get("Close", df.iloc[:, 0])
                )
                out = pd.DataFrame(index=df.index)
                out["Close"] = close
                out["Open"] = close.shift(1).fillna(close)
                out["High"] = out[["Open", "Close"]].max(axis=1) * 1.002
                out["Low"] = out[["Open", "Close"]].min(axis=1) * 0.998
                out["Volume"] = 1_000_000
                return out[["Open", "High", "Low", "Close", "Volume"]]
            return df[["Open", "High", "Low", "Close", "Volume"]]
        
        # If no offline data, raise error in offline mode
        raise RuntimeError(f"No offline data available for {symbol}. Expected fixture at {fixture_path} or cache at {cache_path}")
    
    # Online mode: use yfinance
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required for this script when not in offline mode") from e
    
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"No data for {symbol} in range {start}..{end}")
    # Ensure UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def choose_fold_params(n: int, target_folds: int) -> tuple[int, int, int]:
    # Simple heuristic: 70% train / 30% test over target_folds, rolling by test
    test_len = max(30, n // (target_folds * 3)) if n < target_folds * 100 else max(63, n // (target_folds * 3))
    train_len = max(test_len * 2, 126)
    stride = test_len
    return train_len, test_len, stride


def annualize_cagr(total_return: float, bars: int) -> float:
    if bars <= 0:
        return 0.0
    try:
        return (1.0 + total_return) ** (252.0 / float(bars)) - 1.0
    except Exception:
        return 0.0


def run_for_symbol_profile(
    symbol: str,
    start: str,
    end: str,
    target_folds: int,
    validate_data: bool,
    train_days: int | None = None,
    test_days: int | None = None,
    warmup_days: int = 60,
) -> tuple[list[dict], list[int], int]:
    data = load_data(symbol, start, end)
    X, y, prices = build_feature_table(data, warmup_days=warmup_days)
    n = len(X)
    if train_days is not None and test_days is not None:
        train_len, test_len = int(train_days), int(test_days)
        stride = test_len
    else:
        train_len, test_len, stride = choose_fold_params(n, target_folds)
    folds = list(
        gen_walkforward(
            n=n,
            train_len=train_len,
            test_len=test_len,
            stride=stride,
            warmup=0,
            anchored=False,
        )
    )
    logging.info(
        "walkforward: start=%s end=%s train=%sd test=%sd folds=%s symbol=%s",
        start,
        end,
        train_len,
        test_len,
        len(folds),
        symbol,
    )
    if len(folds) == 0:
        raise ValueError(
            f"Computed 0 folds (n={n}) with train_len={train_len}, test_len={test_len}, stride={stride}. Increase date range or adjust windows."
        )
    if len(folds) == 0:
        raise ValueError(
            f"Computed 0 folds (n={n}) with train_len={train_len}, test_len={test_len}, stride={stride}. Increase date range or adjust windows."
        )
    if len(folds) > target_folds:
        folds = folds[:target_folds]

    pipeline = LeakageProofPipeline(X, y)
    results = walkforward_run(
        pipeline,
        folds,
        prices,
        performance_mode="RELAXED",
        validate_data=validate_data,
    )
    # Attach fold lengths for CAGR
    fold_test_lengths = [f.test_hi - f.test_lo + 1 for f in folds]
    total_trades = 0
    for _, _, trades in results:
        if trades and isinstance(trades, list) and trades[0].get("count") is not None:
            total_trades += int(trades[0]["count"])
    return results, fold_test_lengths, total_trades


def run_smoke(symbols: list[str], train: int = 60, test: int = 10) -> dict:
    # Use fixed short window and CI data if present
    start, end = "2020-01-01", "2020-03-31"
    ordered = symbols
    # Fast pre-check: DataSanity in enforce mode for CI
    sanity_profile = "walkforward_ci" if in_ci() else "walkforward"
    if DataSanityValidator is not None:
        try:
            validator = DataSanityValidator("config/data_sanity.yaml", profile=sanity_profile)
            for sym in ordered:
                df_sanity = load_data(sym, start, end)
                res = validator.validate_dataframe_fast(df_sanity, sanity_profile)
                if res.violations and getattr(res, "mode", "warn") == "enforce":
                    code, reason = _map_violation(res)
                    return {
                        "status": "FAIL",
                        "violation_code": code,
                        "reason": reason,
                        "symbols": [sym],
                        "folds": 0,
                        "any_nan_inf": False,
                        "total_trades": 0,
                    }
        except Exception:
            # Do not fail pre-check hard; main run will still guard
            pass
    start_time = time.monotonic()
    summary = make_report(
        out_path=Path("docs/analysis")
        / f"walkforward_smoke_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}.md",
        profiles=["risk_balanced"],
        symbols=ordered,
        start=start,
        end=end,
        validate_data=True,
        target_folds=1,
        train_days=train,
        test_days=test,
    )
    duration = time.monotonic() - start_time
    return {
        "status": "OK",
        "folds": 1,
        "total_trades": int(summary.get("total_trades", 0)),
        "any_nan_inf": bool(summary.get("any_nan_inf", False)),
        "duration_s": float(f"{duration:.3f}"),
        "symbols": ordered,
    }


def make_report(
    out_path: Path,
    profiles: list[str],
    symbols: list[str],
    start: str,
    end: str,
    validate_data: bool,
    target_folds: int,
    train_days: int | None,
    test_days: int | None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(
        f"# Walkforward Report — {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n"
    )
    lines.append(f"Date range: {start} → {end}\n")
    lines.append(f"Symbols: {', '.join(symbols)}\n")
    lines.append(f"Profiles: {', '.join(profiles)}\n")
    lines.append(f"DataSanity: {'ON' if validate_data else 'OFF'}\n")

    overall_summary: dict[str, dict[str, float]] = {}
    any_nan_inf = False
    total_trades_all = 0

    for profile in profiles:
        lines.append(f"\n## Profile: {profile}\n")
        lines.append(
            "| Symbol | Fold | Sharpe | MaxDD | HitRate | Return | MedianHold | CAGR est. | Risk OK |\n"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|\n")

        sharpe_vals: list[float] = []
        maxdd_vals: list[float] = []

        for symbol in symbols:
            results, fold_lengths, symbol_trades = run_for_symbol_profile(
                symbol=symbol,
                start=start,
                end=end,
                target_folds=target_folds,
                validate_data=validate_data,
                train_days=train_days,
                test_days=test_days,
                warmup_days=10
                if (
                    train_days is not None
                    and test_days is not None
                    and (train_days <= 30 or test_days <= 10)
                )
                else 60,
            )
            total_trades_all += symbol_trades
            for (fold_id, metrics, _), bars in zip(results, fold_lengths, strict=False):
                sharpe = float(metrics.get("sharpe_nw", 0.0))
                max_dd = float(metrics.get("max_dd", 0.0))
                hit_rate = float(metrics.get("hit_rate", 0.0))
                total_return = float(metrics.get("total_return", 0.0))
                median_hold = float(metrics.get("median_hold", 0.0))
                cagr_est = annualize_cagr(total_return, bars)
                risk_ok = max_dd >= -0.03
                if not all(math.isfinite(x) for x in [sharpe, max_dd, total_return, cagr_est]):
                    any_nan_inf = True

                sharpe_vals.append(sharpe)
                maxdd_vals.append(max_dd)

                lines.append(
                    f"| {symbol} | {fold_id} | {sharpe:.3f} | {max_dd:.3f} | {hit_rate:.2f} | "
                    f"{total_return:.3f} | {median_hold:.0f} | {cagr_est:.3f} | "
                    f"{'✅' if risk_ok else '❌'} |\n"
                )

        # Profile summary
        avg_sharpe = float(np.mean(sharpe_vals)) if sharpe_vals else 0.0
        avg_maxdd = float(np.mean(maxdd_vals)) if maxdd_vals else 0.0
        overall_summary[profile] = {"avg_sharpe": avg_sharpe, "avg_maxdd": avg_maxdd}
        lines.append("\n")
        lines.append(f"- Average Sharpe: {avg_sharpe:.3f}\n")
        lines.append(f"- Average MaxDD: {avg_maxdd:.3f}\n")

    # Recommendation
    best_profile = None
    best_score = -1e9
    for p, s in overall_summary.items():
        score = s["avg_sharpe"] - abs(min(0.0, s["avg_maxdd"])) * 2.0
        if score > best_score:
            best_score = score
            best_profile = p
    lines.append("\n### Recommendation\n")
    if best_profile is None:
        lines.append("No clear winner.\n")
    else:
        lines.append(f"Recommend profile: {best_profile} (balance of Sharpe and drawdown).\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    return {
        "overall_summary": overall_summary,
        "any_nan_inf": any_nan_inf,
        "total_trades": total_trades_all,
    }


def _normalize_symbols(raw: list[str]) -> list[str]:
    combined: list[str] = []
    for token in raw:
        if "," in token:
            combined.extend([t for t in token.split(",") if t])
        else:
            combined.append(token)
    seen = set()
    out: list[str] = []
    for s in combined:
        sx = s.strip()
        if not sx:
            continue
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def main():
    ap = argparse.ArgumentParser(description="Multi-profile walkforward report")
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "BTC-USD", "TSLA"],
        help="Symbols to analyze",
    )
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=["risk_low", "risk_balanced", "risk_strict"],
    )
    ap.add_argument(
        "--separate-crypto", action="store_true", help="Run crypto in a separate section"
    )
    ap.add_argument("--validate-data", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--train-window", type=int, default=None)
    ap.add_argument("--test-window", type=int, default=None)
    ap.add_argument("--max-runtime", type=int, default=60)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    # Determinism
    random.seed(1337)
    np.random.seed(1337)
    try:
        import torch  # type: ignore

        torch.manual_seed(1337)
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True)  # type: ignore
    except Exception:
        pass

    # Logging level
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))

    # Resolve effective params
    symbols = _normalize_symbols(args.symbols)
    if args.smoke and not symbols:
        symbols = ["SPY", "BTC-USD"]
    if not symbols:
        raise ValueError("No symbols provided.")
    if args.smoke:
        eff_start = "2020-01-01"
        eff_end = "2020-03-31"
        eff_folds = 1
        eff_train = 60
        eff_test = 10
        out_prefix = "walkforward_smoke"
    else:
        eff_start = args.start
        eff_end = args.end
        eff_folds = args.folds or 5
        eff_train = args.train_window
        eff_test = args.test_window
        out_prefix = "walkforward"

    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    out_file = Path("docs/analysis") / f"{out_prefix}_{ts}.md"

    # Separate crypto vs. non-crypto if requested
    crypto = [s for s in symbols if s.endswith("-USD")]
    non_crypto = [s for s in symbols if s not in crypto]

    if args.separate_crypto and crypto and non_crypto:
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_non = out_file.with_name(out_file.stem + f"_noncrypto_{ts}.md")
        out_cry = out_file.with_name(out_file.stem + f"_crypto_{ts}.md")
        make_report(
            out_non,
            args.profiles,
            non_crypto,
            eff_start,
            eff_end,
            args.validate_data,
            eff_folds,
            eff_train,
            eff_test,
        )
        make_report(
            out_cry,
            args.profiles,
            crypto,
            eff_start,
            eff_end,
            args.validate_data,
            eff_folds,
            eff_train,
            eff_test,
        )
        print(f"Report written: {out_non}\nReport written: {out_cry}")
    else:
        ordered = non_crypto + crypto
        # For smoke, ensure ≥1 fold by adapting windows if needed
        if args.smoke:
            probe_symbol = ordered[0]
            data_probe = load_data(probe_symbol, eff_start, eff_end)
            # Use small warmup to maximize available bars
            X_probe, _, _ = build_feature_table(data_probe, warmup_days=10)
            n_probe = len(X_probe)
            if eff_train + eff_test > n_probe:
                # Adapt windows conservatively
                new_train = max(10, min(30, n_probe - (eff_test or 10) - 1))
                new_test = max(5, min(10, n_probe - new_train - 1))
                logging.info(
                    "smoke: adapting windows train %s->%s, test %s->%s to allow ≥1 fold (n=%s)",
                    eff_train,
                    new_train,
                    eff_test,
                    new_test,
                    n_probe,
                )
                eff_train, eff_test = new_train, new_test
        start_time = time.monotonic()
        summary = make_report(
            out_path=out_file,
            profiles=args.profiles,
            symbols=ordered,
            start=eff_start,
            end=eff_end,
            validate_data=args.validate_data,
            target_folds=eff_folds,
            train_days=eff_train,
            test_days=eff_test,
        )
        # Smoke guardrails and summary
        if args.smoke:
            try:
                if summary.get("any_nan_inf"):
                    payload = {
                        "status": "FAIL",
                        "violation_code": "NAN_INF_METRICS",
                        "reason": "NaN/inf metrics detected",
                        "symbols": ordered,
                    }
                    _write_smoke_json(payload)
                    raise SystemExit(1)
                if int(summary.get("total_trades", 0)) <= 0:
                    payload = {
                        "status": "FAIL",
                        "violation_code": "NO_TRADES",
                        "reason": "No trades across test window",
                        "symbols": ordered,
                    }
                    _write_smoke_json(payload)
                    raise SystemExit(1)
                duration = time.monotonic() - start_time
                if duration > args.max_runtime:
                    payload = {
                        "status": "FAIL",
                        "violation_code": "RUNTIME_BUDGET",
                        "reason": f"Exceeded {args.max_runtime}s",
                        "symbols": ordered,
                    }
                    _write_smoke_json(payload)
                    raise SystemExit(1)
                overall = summary.get("overall_summary", {})
                avg_sharpe = (
                    float(np.mean([v.get("avg_sharpe", 0.0) for v in overall.values()]))
                    if overall
                    else 0.0
                )
                avg_maxdd = (
                    float(np.mean([v.get("avg_maxdd", 0.0) for v in overall.values()]))
                    if overall
                    else 0.0
                )
                print(
                    f"SMOKE OK | folds={eff_folds} | symbols={','.join(ordered)} | sharpe={avg_sharpe:.3f} maxdd={avg_maxdd:.3f}"
                )
                # Build per-fold summaries (use average placeholders if detailed not available from framework)
                fold_summaries = []
                for idx_fold in range(eff_folds):
                    fold_summaries.append(
                        {
                            "fold": idx_fold,
                            "sharpe": float(avg_sharpe),
                            "max_drawdown": float(avg_maxdd),
                            "trades": int(summary.get("total_trades", 0)),
                            "duration_ms": int(duration * 1000 / max(eff_folds, 1)),
                        }
                    )
                # Load risk costs from config for provenance/visibility
                risk_costs = {}
                try:
                    with open("config/base.yaml", encoding="utf-8") as f:
                        cfg_yaml = yaml.safe_load(f)
                        risk_cfg = (cfg_yaml or {}).get("risk", {})
                        risk_costs = {
                            "slippage_bps": risk_cfg.get("slippage_bps"),
                            "fee_bps": risk_cfg.get("fee_bps"),
                            "max_leverage": risk_cfg.get("max_leverage"),
                        }
                except Exception:
                    risk_costs = {}

                payload_ok = {
                    "status": "OK",
                    "folds": eff_folds,
                    "symbols": ordered,
                    "trades": int(summary.get("total_trades", 0)),
                    "duration_s": float(f"{duration:.3f}"),
                    "sharpe": float(avg_sharpe),
                    "max_drawdown": float(avg_maxdd),
                    "fold_summaries": fold_summaries,
                    "risk_costs": risk_costs,
                    "seed": 1337,
                    "auto_adjust": False,
                }
                _write_smoke_json(payload_ok)
                write_provenance("reports/smoke_provenance.json", ["config/base.yaml"])
                _write_smoke_meta(payload_ok)
                _write_run_json({"timestamp": payload_ok.get("run_id")})
            except SystemExit:
                raise
            except Exception as e:
                fail_payload = {
                    "status": "FAIL",
                    "violation_code": "UNEXPECTED_ERROR",
                    "reason": f"{e.__class__.__name__}: {e}",
                    "symbols": ordered,
                }
                _write_smoke_json(fail_payload)
                raise

    print(f"Report written: {out_file}")


if __name__ == "__main__":
    main()
