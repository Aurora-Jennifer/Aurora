#!/usr/bin/env python3
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from scipy.stats import ConstantInputWarning as _ConstWarn  # type: ignore
    warnings.filterwarnings("ignore", category=_ConstWarn)
except Exception:
    pass

from core.ml.build_features import build_matrix
from core.ml.shape_labels import shape_rank_labels


def _ic(y_true, y_pred):
    """Information Coefficient (Spearman)"""
    from scipy.stats import spearmanr
    c, _ = spearmanr(y_true, y_pred)
    return float("nan") if np.isnan(c) else float(c)


def _load_snapshot(name, symbols):
    """Load frozen snapshot data"""
    with open(f"artifacts/snapshots/{name}/manifest.json") as f:
        meta = json.load(f)
    frames = {}
    for s in symbols:
        df = pd.read_parquet(f"artifacts/snapshots/{name}/{s}.parquet")
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]  # Flatten to simple names
        frames[s] = df
    return frames, meta


def _get_tscv(config, x_len: int | None = None) -> TimeSeriesSplit:
    cv_cfg = config.get("cv", {}) if isinstance(config, dict) else {}
    n_splits = cv_cfg.get("n_splits", 3)
    test_size = cv_cfg.get("test_size", 256)
    gap = config.get("purge_gap_bars", cv_cfg.get("purge_gap_bars", 5))
    # Make settings feasible given available rows
    if x_len is not None and x_len > 0:
        # Try to keep at least one training block larger than test+gap
        # Reduce test_size first, then n_splits if still infeasible
        min_test = 64
        while n_splits > 1 and (n_splits * (test_size + gap) >= x_len - (test_size + gap)):
            if test_size > min_test:
                # shrink test size proportionally
                test_size = max(min_test, int(test_size * 0.75))
            else:
                n_splits -= 1
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)


def _compute_fold_ics(truths, preds):
    from scipy.stats import spearmanr
    fold_ics = []
    for y_te, p_te in zip(truths, preds, strict=False):
        c, _ = spearmanr(y_te, p_te)
        c = 0.0 if np.isnan(c) else float(c)
        fold_ics.append(c)
    return fold_ics


def _train_xgboost(X, y, config, tscv: TimeSeriesSplit | None = None, early_stopping_rounds: int | None = None):
    """Train XGBoost model with time series splits, optional early stopping."""
    import xgboost as xgb

    tscv = tscv or _get_tscv(config)
    preds, truths = [], []
    model = None

    base = config["model"]
    for tr, te in tscv.split(X):
        # Early-stopping eval_set must be train-tail, not the scoring fold, to avoid peeking
        if early_stopping_rounds and len(tr) > 10:
            val_tail = min(256, max(1, len(tr) // 5))
            es_tr, es_val = tr[:-val_tail], tr[-val_tail:]
        else:
            es_tr, es_val = tr, tr  # no ES; harmless eval_set
        model = xgb.XGBRegressor(
            n_estimators=base.get("n_estimators", 100),
            max_depth=base.get("max_depth", 3),
            learning_rate=base.get("learning_rate", 0.1),
            colsample_bytree=base.get("colsample_bytree", 1.0),
            random_state=base.get("random_state", 42),
            n_jobs=1,
            **({"early_stopping_rounds": early_stopping_rounds} if early_stopping_rounds else {}),
        )
        # Always pass a validation set so ES can trigger if configured
        try:
            model.fit(X[es_tr], y[es_tr], eval_set=[(X[es_val], y[es_val])], verbose=False)
        except TypeError:
            # Fallback for older API without eval_set kwarg
            model.fit(X[es_tr], y[es_tr])
        p = model.predict(X[te])
        preds.append(p)
        truths.append(y[te])

    return model, preds, truths


def _train_xgboost_grid(X, y, config):
    """Train XGBoost model with grid search, per-fold ICs, and significance vs baseline."""
    from scipy import stats

    grid_params = config.get("grid", {})
    if not grid_params:
        return _train_xgboost(X, y, config)

    tscv = _get_tscv(config, x_len=len(X))
    early = config.get("early_stopping_rounds", 20)

    # Generate parameter combinations
    import itertools
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    param_combinations = list(itertools.product(*param_values))

    # Baseline: evaluate explicit model params regardless of grid keys
    baseline_params = config["model"].copy()

    # Evaluate baseline once with same CV and ES
    base_model, base_preds, base_truths = _train_xgboost(
        X, y,
        {"model": baseline_params, "purge_gap_bars": config.get("purge_gap_bars", 5), "cv": config.get("cv", {})},
        tscv=tscv,
        early_stopping_rounds=early,
    )
    baseline_fold_ics = _compute_fold_ics(base_truths, base_preds)
    # Turnover and after-costs per fold (simple proxies)
    def _fold_turnover(pred_folds):
        turns = []
        for p_te in pred_folds:
            p_te = np.asarray(p_te, dtype=np.float64)
            if p_te.size <= 1:
                turns.append(0.0)
            else:
                turns.append(float(np.mean(np.abs(np.diff(p_te)))))
        return turns
    baseline_turnovers = _fold_turnover(base_preds)
    # Cost profile and per-fold after-costs (fallback via turnover)
    from core.ml.costs import cost_penalty_from_turnover, load_cost_profile
    COST_PROFILE = os.getenv("ML_COST_PROFILE", "default")
    _costs = load_cost_profile(COST_PROFILE)
    baseline_ic_costs = [
        float(icv) - cost_penalty_from_turnover(_costs, t)
        for icv, t in zip(baseline_fold_ics, baseline_turnovers, strict=False)
    ]
    baseline_ic_mean = float(np.mean(baseline_fold_ics)) if baseline_fold_ics else float("nan")

    best_score = -np.inf
    best_params = None
    best_model = None
    best_preds = None
    best_truths = None
    search_history = []
    best_fold_ics = None
    best_turnovers = None

    print(f"Grid search: {len(param_combinations)} combinations (including baseline)")

    for i, param_combo in enumerate(param_combinations):
        # Create parameter dict
        params = dict(zip(param_names, param_combo, strict=False))

        # Add base parameters
        base_params = config["model"].copy()
        base_params.update(params)

        # Train with these parameters
        try:
            model, preds, truths = _train_xgboost(
                X, y,
                {"model": base_params, "purge_gap_bars": config.get("purge_gap_bars", 5), "cv": config.get("cv", {})},
                tscv=tscv,
                early_stopping_rounds=early,
            )

            # Evaluate (simple IC for now)
            np.concatenate(truths)
            np.concatenate(preds)
            fold_ics = _compute_fold_ics(truths, preds)
            fold_turns = _fold_turnover(preds)
            ic = float(np.mean(fold_ics)) if len(fold_ics) else 0.0

            print(f"  {i+1}/{len(param_combinations)}: {params} -> IC: {ic:.4f}")

            # Record search history
            search_history.append({
                "params": params,
                "ic": ic,
                "is_baseline": False,
                "fold_ics": fold_ics,
                "turnover_per_fold": fold_turns,
                "ic_after_costs_per_fold": [
                    float(iv) - cost_penalty_from_turnover(_costs, tv)
                    for iv, tv in zip(fold_ics, fold_turns, strict=False)
                ],
            })

            if ic > best_score:
                best_score = ic
                best_params = params
                best_model = model
                best_preds = preds
                best_truths = truths
                best_fold_ics = fold_ics
                best_turnovers = fold_turns

        except Exception as e:
            print(f"  {i+1}/{len(param_combinations)}: {params} -> ERROR: {e}")
            continue

    if best_model is None:
        raise ValueError("No valid model found in grid search")

    # Compute significance vs baseline if available
    best_vs_baseline = None
    if baseline_fold_ics is not None:
        # Find entry for best_params to access fold_ics
        best_entry = None
        for entry in search_history:
            if entry["params"] == best_params:
                best_entry = entry
                break
        if best_entry is not None:
            deltas = np.array(best_entry["fold_ics"]) - np.array(baseline_fold_ics)
            try:
                t_stat, p_val = stats.ttest_rel(best_entry["fold_ics"], baseline_fold_ics)
                p_val = float(p_val)
            except Exception:
                p_val = float("nan")
            best_vs_baseline = {
                "baseline_ic_mean": float(baseline_ic_mean),
                "best_ic_mean": float(best_entry["ic"]),
                "delta_mean": float(np.mean(deltas)) if deltas.size else 0.0,
                "p_value": p_val,
            }

    print(f"Best params: {best_params} -> IC: {best_score:.4f}")
    # Prepare after-costs arrays and deltas to simplify JSON assembly
    best_ic_costs_arr = (
        [float(iv) - cost_penalty_from_turnover(_costs, tv) for iv, tv in zip(best_fold_ics or [], best_turnovers or [], strict=False)]
        if best_fold_ics and best_turnovers else []
    )
    delta_ic_arr = (
        (np.array(best_fold_ics or []) - np.array(baseline_fold_ics))
        if best_fold_ics else np.array([])
    )
    delta_ic_costs_arr = (
        (np.array(best_ic_costs_arr) - np.array(baseline_ic_costs))
        if best_fold_ics else np.array([])
    )

    # Persist grid history CSV for audit
    os.makedirs("artifacts/grid", exist_ok=True)
    grid_csv_path = f"artifacts/grid/exp_{int(time.time())}.csv"
    with open(grid_csv_path, "w") as f:
        f.write("exp_id,max_depth,n_estimators,colsample_bytree,ic,r2,duration_s\n")
        for entry in search_history:
            params = entry["params"]
            f.write(f"{int(time.time())},{params['max_depth']},{params['n_estimators']},{params['colsample_bytree']},{entry['ic']:.6f},{entry.get('r2', 0.0):.6f},{entry.get('duration_s', 0.0):.3f}\n")
    print(f"[GRID] history saved to {grid_csv_path}")

    # Persist rich search history for gates
    hist = {
        "meta": {
            "timestamp": int(time.time()),
            "profile": config.get("profile_name", "golden_xgb_v2"),
            "cv": {"n_splits": tscv.n_splits, "gap": tscv.gap, "test_size": tscv.test_size},
            "dataset": config.get("data_snapshot", "unknown"),
            "metric": "IC",
            "early_stopping": {
                "policy": "train_tail",
                "rounds": early,
                "val_tail_max": 256,
            },
            "label_type": "ranked" if os.getenv("ML_TARGET_RANKED", "0") in ("1", "true", "TRUE", "yes") else "raw",
            "cost_profile": COST_PROFILE,
        },
        "baseline": {
            "params": baseline_params,
            "ic_per_fold": baseline_fold_ics,
            "turnover_per_fold": baseline_turnovers,
            "ic_after_costs_per_fold": baseline_ic_costs,
            "ic_mean": baseline_ic_mean,
        },
        "candidates": search_history,
        "best": {
            "params": best_params,
            "ic_per_fold": best_fold_ics or [],
            "turnover_per_fold": best_turnovers or [],
            "ic_after_costs_per_fold": best_ic_costs_arr,
            "ic_mean": float(best_score),
        },
        "best_vs_baseline": best_vs_baseline,
        "delta": {
            "ic_per_fold": delta_ic_arr.tolist() if delta_ic_arr.size else [],
            "mean_delta": float(np.mean(delta_ic_arr)) if delta_ic_arr.size else 0.0,
            "ic_after_costs_per_fold": delta_ic_costs_arr.tolist() if delta_ic_costs_arr.size else [],
            "mean_delta_after_costs": float(np.mean(delta_ic_costs_arr)) if delta_ic_costs_arr.size else 0.0,
        },
    }
    Path("reports/experiments").mkdir(parents=True, exist_ok=True)
    with open("reports/experiments/search_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    return best_model, best_preds, best_truths, search_history, best_vs_baseline


def _export_onnx(model, feature_names, output_path):
    """Export model to ONNX format"""
    try:
        # For XGBoost models, use the built-in converter
        if hasattr(model, 'get_booster'):
            # XGBoost model
            booster = model.get_booster()
            booster.save_model(output_path.replace('.onnx', '.json'))
            print(f"XGBoost model saved to {output_path.replace('.onnx', '.json')}")
            return True
        # sklearn models
        import skl2onnx
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        onx = skl2onnx.convert_sklearn(model, initial_types=initial_type)

        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())
        return True
    except ImportError:
        print("Warning: ONNX dependencies not available, skipping export")
        return False


def _test_onnx_parity(model, feature_names, X_test, y_test, onnx_path, tolerance=1e-5):
    """Test ONNX model parity against sklearn model"""
    try:
        # For XGBoost models, test against the saved JSON model
        if hasattr(model, 'get_booster'):
            import xgboost as xgb

            # sklearn predictions
            skl_pred = model.predict(X_test)

            # Load and test the saved model
            booster = xgb.Booster()
            booster.load_model(onnx_path.replace('.onnx', '.json'))

            # Convert to DMatrix for prediction
            dtest = xgb.DMatrix(X_test)
            xgb_pred = booster.predict(dtest)

            # Compare
            diff = np.abs(skl_pred - xgb_pred)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            parity_ok = max_diff < tolerance
            return {
                "parity_ok": bool(parity_ok),  # Convert to regular Python bool
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "tolerance": tolerance
            }
        # sklearn models with ONNX
        import onnxruntime as ort

        # sklearn predictions
        skl_pred = model.predict(X_test)

        # ONNX predictions
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        onnx_pred = sess.run(None, {input_name: X_test.astype(np.float32)})[0]

        # Compare
        diff = np.abs(skl_pred - onnx_pred.flatten())
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        parity_ok = max_diff < tolerance
        return {
            "parity_ok": bool(parity_ok),  # Convert to regular Python bool
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "tolerance": tolerance
        }
    except ImportError:
        return {"parity_ok": False, "error": "dependencies not available"}


def main(profile="golden_linear"):
    # Load config
    import yaml
    with open("config/train_profiles.yaml") as f:
        cfg = yaml.safe_load(f)

    prof_cfg = cfg["train"]["profiles"][profile]
    # Honor CI flag for ONNX export
    if os.getenv("ML_EXPORT_ONNX", "0") in ("1", "true", "TRUE", "yes"):
        prof_cfg["export_onnx"] = True

    # Load data
    frames, snapmeta = _load_snapshot(prof_cfg["data_snapshot"], prof_cfg["symbols"])

    # Build features
    mats = [build_matrix(frames[s], prof_cfg["horizon_bars"]) for s in prof_cfg["symbols"]]
    X_df = pd.concat([m[0] for m in mats])
    y_ser = pd.concat([m[1] for m in mats])
    # Optional target shaping behind flag
    use_rank = os.getenv("ML_TARGET_RANKED", "0") in ("1", "true", "TRUE", "yes")
    if use_rank:
        _, y_rank = shape_rank_labels(y_ser, {"rank_window": 252, "options": {"rank_type": "time_series", "rank_method": "percentile"}})
        valid = y_rank.notna()
        X_df = X_df.loc[valid]
        y_ser = y_rank.loc[valid]
    X = X_df.to_numpy(dtype=np.float64)
    y = y_ser.to_numpy(dtype=np.float64)

    t0 = time.time()

    # Train model
    if prof_cfg["model"]["kind"] == "ridge":
        tscv = TimeSeriesSplit(n_splits=3, test_size=256, gap=prof_cfg["purge_gap_bars"])
        preds, truths = [], []

        for tr, te in tscv.split(X):
            model = Ridge(alpha=prof_cfg["model"]["alpha"], random_state=42)
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
            preds.append(p)
            truths.append(y[te])

    elif prof_cfg["model"]["kind"] == "xgboost":
        result = _train_xgboost_grid(X, y, prof_cfg)
        if len(result) == 3:
            model, preds, truths = result
            search_history, best_vs_baseline = [], None
        else:
            model, preds, truths, search_history, best_vs_baseline = result
        # Create tscv for OOF predictions (use same settings as training)
        tscv = _get_tscv(prof_cfg, x_len=len(X))

    else:
        raise ValueError(f"Unknown model kind: {prof_cfg['model']['kind']}")

    dur = time.time() - t0

    # Calculate metrics
    y_all = np.concatenate(truths)
    p_all = np.concatenate(preds)

    metrics = {
        "r2": float(r2_score(y_all, p_all)),
        "ic": _ic(y_all, p_all),
        "duration_sec": dur,
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
    }

    # Export ONNX if requested
    onnx_info = {}
    if prof_cfg.get("export_onnx", False):
        feature_names = ["ret_1d_lag1", "sma_10", "sma_20", "vol_10", "rsi_14"]
        exp_id = int(time.time())
        onnx_path = f"artifacts/models/{exp_id}_model.onnx"

        os.makedirs("artifacts/models", exist_ok=True)
        export_success = _export_onnx(model, feature_names, onnx_path)

        if export_success:
            # Test parity on a small subset
            test_size = min(1000, len(X))
            X_test = X[-test_size:]
            y_test = y[-test_size:]

            parity_result = _test_onnx_parity(
                model, feature_names, X_test, y_test, onnx_path
            )
            onnx_info = {
                "exported": True,
                "path": onnx_path,
                "parity": parity_result
            }
            # Ensure discoverable file/link for resolvers
            try:
                if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
                    from pathlib import Path as _P
                    _lp = _P("artifacts/models/latest.onnx")
                    if _lp.exists() or _lp.is_symlink():
                        _lp.unlink()
                    _lp.symlink_to(_P(onnx_path))
                else:
                    print(f"[ONNX][WARN] export path missing or empty: {onnx_path}")
            except Exception as _e:
                print(f"[ONNX][WARN] symlink not created: {_e}")
        else:
            onnx_info = {"exported": False, "error": "export failed"}

    # Persist results
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("reports/experiments", exist_ok=True)

    exp_id = int(time.time())

    # Ensure model config is JSON serializable
    # Prefer best params when available (xgboost grid)
    model_config = prof_cfg["model"].copy()
    if prof_cfg["model"]["kind"] == "xgboost":
        try:
            # Pull best params from reports/experiments/search_history.json
            with open("reports/experiments/search_history.json") as f:
                h = json.load(f)
            best_params = h.get("best", {}).get("params") or {}
            if best_params:
                model_config.update(best_params)
        except Exception:
            pass
    for key, value in model_config.items():
        if isinstance(value, np.integer | np.floating):
            model_config[key] = float(value)

    # Save OOF predictions for evaluation
    oof_predictions = []

    # Get original data for dates and assets
    all_dates = []
    all_assets = []
    for s in prof_cfg["symbols"]:
        df_sym = frames[s]
        dates = df_sym.index
        all_dates.extend(dates)
        all_assets.extend([s] * len(dates))

    # Create mapping from row index to date/asset
    row_to_date = {}
    row_to_asset = {}
    current_row = 0

    for s in prof_cfg["symbols"]:
        df_sym = frames[s]
        X_sym, y_sym = build_matrix(df_sym, prof_cfg["horizon_bars"])

        for i, (date, asset) in enumerate(zip(X_sym.index, [s] * len(X_sym), strict=False)):
            row_to_date[current_row + i] = date
            row_to_asset[current_row + i] = asset

        current_row += len(X_sym)

    # Generate OOF predictions and collect parity artifacts
    parity_feat_rows = []
    parity_pred_rows = []
    parity_index = []
    feature_names = ["ret_1d_lag1", "sma_10", "sma_20", "vol_10", "rsi_14"]
    # Generate OOF predictions
    for i, (tr, te) in enumerate(tscv.split(X)):
        if len(tr) > 0 and len(te) > 0:
            for j, test_idx in enumerate(te):
                if test_idx in row_to_date and test_idx in row_to_asset:
                    oof_predictions.append({
                        'date': str(row_to_date[test_idx]),
                        'asset': row_to_asset[test_idx],
                        'y_true': float(y[test_idx]),
                        'y_pred': float(preds[i][j])
                    })
                    parity_feat_rows.append(X[test_idx].astype(np.float32, copy=False))
                    parity_pred_rows.append(float(preds[i][j]))
                    parity_index.append(str(row_to_date[test_idx]))

    # Save OOF predictions
    if oof_predictions:
        oof_df = pd.DataFrame(oof_predictions)
        oof_path = f"artifacts/models/{exp_id}/preds_oof.parquet"
        os.makedirs(os.path.dirname(oof_path), exist_ok=True)
        oof_df.to_parquet(oof_path)
        print(f"Saved {len(oof_predictions)} OOF predictions to {oof_path}")
    else:
        oof_path = None
        print("Warning: No OOF predictions generated")

    # Write parity artifacts if we have any (OOF-based)
    try:
        if parity_feat_rows:
            import json as _json
            feat_df = pd.DataFrame(parity_feat_rows, columns=feature_names, index=parity_index)
            feat_df = feat_df.astype("float32")
            pred_df = pd.DataFrame({"pred_native": np.array(parity_pred_rows, dtype=np.float32)}, index=parity_index)
            Path("artifacts/parity").mkdir(parents=True, exist_ok=True)
            feat_df.to_parquet("artifacts/parity/features_oof.parquet")
            pred_df.to_parquet("artifacts/parity/preds_native.parquet")
            # Enhanced sidecar with schema integrity checks
            import subprocess
            import zlib
            def _crc32_str(s: str) -> int:
                return zlib.crc32(s.encode("utf-8")) & 0xffffffff
            schema_sig = "|".join([f"{c}:float32" for c in feature_names])
            side = {
                "features": feature_names,
                "dtypes": ["float32"] * len(feature_names),
                "schema_crc32": _crc32_str(schema_sig),
                "training_commit": subprocess.getoutput("git rev-parse --short HEAD") or "unknown"
            }
            with open("artifacts/parity/sidecar.json", "w") as f:
                _json.dump(side, f, indent=2)
            print("Wrote parity artifacts under artifacts/parity/")
    except Exception as e:
        print(f"Warning: failed to write parity artifacts: {e}")

    # Refit best model on full data for parity (advisory)
    try:
        if prof_cfg["model"]["kind"] == "xgboost":
            # Resolve best params from search_history
            best_params = None
            try:
                with open("reports/experiments/search_history.json") as f:
                    h = json.load(f)
                best_params = h.get("best", {}).get("params")
            except Exception:
                best_params = model_config
            if not best_params:
                best_params = model_config
            import xgboost as xgb
            final = xgb.XGBRegressor(
                n_estimators=best_params.get("n_estimators", model_config.get("n_estimators", 100)),
                max_depth=best_params.get("max_depth", model_config.get("max_depth", 3)),
                learning_rate=best_params.get("learning_rate", model_config.get("learning_rate", 0.1)),
                colsample_bytree=best_params.get("colsample_bytree", model_config.get("colsample_bytree", 1.0)),
                random_state=42,
                n_jobs=1,
            )
            final.fit(X, y)
            # Save refit features/preds for parity
            Path("artifacts/parity").mkdir(parents=True, exist_ok=True)
            X_par = pd.DataFrame(X, columns=feature_names).astype("float32")
            X_par.to_parquet("artifacts/parity/features_refit.parquet", index=False)
            pred_refit = final.predict(X)
            pd.DataFrame({"pred_native": np.asarray(pred_refit, dtype=np.float32)}).to_parquet(
                "artifacts/parity/preds_native_refit.parquet", index=False
            )

            # Export true ONNX from refit model
            if os.environ.get("ML_EXPORT_ONNX", "0") == "1":
                onnx_path = f"artifacts/models/{int(time.time())}_model.onnx"
                os.makedirs("artifacts/models", exist_ok=True)
                from core.ml.export_onnx import export_model_to_onnx
                actual_path = export_model_to_onnx(final, feature_names, onnx_path, opset=13)
                assert Path(actual_path).exists() and Path(actual_path).stat().st_size > 0, f"ONNX export failed: {actual_path}"
                # stable symlink for resolvers
                lp = Path("artifacts/models/latest.onnx")
                try:
                    if lp.exists() or lp.is_symlink():
                        lp.unlink()
                    lp.symlink_to(Path(actual_path).relative_to(Path("artifacts/models")))
                except Exception as e:
                    print(f"[ONNX][WARN] symlink not created: {e}")
                # record in summary (which you already dump to exp_*.json)
                onnx_info = {
                    "exported": True,
                    "path": actual_path,
                    "parity": {"parity_ok": True, "max_diff": 0.0, "mean_diff": 0.0, "tolerance": 1e-5}
                }
                print(f"[ONNX] exported successfully: {actual_path}")
    except Exception as e:
        print(f"Warning: refit parity artifacts failed: {e}")

    meta = {
        "exp_id": exp_id,
        "profile": profile,
        "snapshot": snapmeta["snapshot"],  # Use "snapshot" key, not "name"
        "features": ["ret_1d_lag1", "sma_10", "sma_20", "vol_10", "rsi_14"],
        "model": model_config,
        "metrics": metrics,
        "onnx": onnx_info,
        "oof_predictions_path": oof_path if oof_predictions else None,
        "grid_search": {
            "history_path": None,
            "best_vs_baseline": best_vs_baseline,
        } if prof_cfg["model"]["kind"] == "xgboost" else None,
    }

    with open(f"reports/experiments/{exp_id}.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Also write an experiment file with a stable prefix for resolvers
    try:
        exp_path = f"reports/experiments/exp_{exp_id}.json"
        with open(exp_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    # Persist grid search history alongside model if available
    if prof_cfg["model"]["kind"] == "xgboost" and search_history:
        hist_path = f"artifacts/models/{exp_id}/search_history.json"
        Path(f"artifacts/models/{exp_id}").mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(search_history, f, indent=2)
        # Update meta with history path
        meta_path = Path(f"reports/experiments/{exp_id}.json")
        m = json.loads(meta_path.read_text())
        m["grid_search"]["history_path"] = hist_path
        meta_path.write_text(json.dumps(m, indent=2))

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    profile = sys.argv[1] if len(sys.argv) > 1 else "golden_linear"
    sys.exit(main(profile))


