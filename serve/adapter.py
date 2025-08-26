#!/usr/bin/env python3
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import yaml  # optional, for costs loading
except Exception:  # pragma: no cover
    yaml = None

import contextlib

from core.ml.build_features import build_matrix

# Configure logging
from core.utils import setup_logging
logger = setup_logging("logs/serve_adapter.log", logging.INFO)


class ModelAdapter:
    """Paper-trade-ready model adapter for real-time predictions"""

    def __init__(self, model_path: str, feature_names: list[str], min_history_bars: int = 20):
        """
        Initialize the model adapter.

        Args:
            model_path: Path to the trained model (.json)
            feature_names: List of feature names in order
            min_history_bars: Minimum bars required for feature calculation
        """
        self.model_path = model_path
        self.feature_names = feature_names
        self.min_history_bars = max(min_history_bars, self._compute_required_history(feature_names))

        # Load model (native XGBoost JSON by default)
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self._onnx_sess = None
        self._predict_fn = self.model.predict
        # Optional ONNX path: try sibling .onnx
        try:
            import onnxruntime as ort  # type: ignore
            onnx_candidate = Path(self.model_path).with_suffix(".onnx")
            if onnx_candidate.exists():
                self._onnx_sess = ort.InferenceSession(str(onnx_candidate), providers=["CPUExecutionProvider"])  # pragma: no cover
                self._onnx_input_name = self._onnx_sess.get_inputs()[0].name
                self._predict_fn = self._predict_via_onnx
                logger.info(f"ONNX session enabled: {onnx_candidate}")
        except Exception as onnx_err:
            logger.info(f"ONNX not enabled ({onnx_err}); falling back to native model")

        # Load model metadata
        self.metadata = self._load_metadata()

        # Initialize history storage
        self.history = {}  # symbol -> DataFrame

        # Warmup the model path
        self._warmup()

        # Telemetry
        self._num_predictions = 0
        self._last_latency_ms: float | None = None
        self.costs = self._load_costs()

        logger.info(f"Model adapter initialized with {len(feature_names)} features")

    def _load_metadata(self) -> dict:
        """Load model metadata from experiment JSON"""
        # Try to find the experiment JSON
        model_dir = Path(self.model_path).parent
        exp_files = list(model_dir.glob("*.json"))

        for exp_file in exp_files:
            if "search_history" not in exp_file.name:
                try:
                    with open(exp_file) as f:
                        metadata = json.load(f)
                    if "features" in metadata:
                        return metadata
                except Exception:
                    continue

        # Fallback metadata
        return {
            "features": self.feature_names,
            "model_path": self.model_path
        }

    def _compute_required_history(self, feature_names: list[str]) -> int:
        """Infer minimum bars needed based on features present.
        Rules assume 1-step shift applied in builder for all rolling features.
        """
        required = 2  # ret_1d_lag1 needs at least 2 rows to be fully valid post shift
        for name in feature_names:
            if name.startswith("sma_"):
                try:
                    window = int(name.split("_")[1])
                    required = max(required, window + 1)
                except Exception:
                    required = max(required, 21)
            if name.startswith("vol_"):
                try:
                    window = int(name.split("_")[1])
                    required = max(required, window + 1)
                except Exception:
                    required = max(required, 11)
            if name.startswith("rsi_"):
                try:
                    window = int(name.split("_")[1])
                    required = max(required, window + 1)
                except Exception:
                    required = max(required, 15)
        return required

    def _warmup(self):
        """Warm up the model with dummy data"""
        dummy_data = np.random.randn(1, len(self.feature_names)).astype(np.float32)
        try:
            _ = self._predict_fn(dummy_data)
        except Exception:
            _ = self.model.predict(dummy_data)
        logger.info("Model warmup completed")

    def _predict_via_onnx(self, features: np.ndarray) -> np.ndarray:  # pragma: no cover
        assert self._onnx_sess is not None
        inputs = {self._onnx_input_name: features.astype(np.float32, copy=False)}
        outputs = self._onnx_sess.run(None, inputs)
        return outputs[0].squeeze(-1)

    def _prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray | None, list[str] | None]:
        """Build latest features, enforce schema and dtype.

        Returns (features_array, error_list). If error_list is not None, preparation failed.
        """
        X, _ = build_matrix(df, horizon=1)
        if len(X) == 0:
            return None, ["empty_features"]

        # Enforce schema: order and presence
        missing = [c for c in self.feature_names if c not in X.columns]
        extra = [c for c in X.columns if c not in self.feature_names]
        if missing or extra:
            return None, [f"missing:{missing}", f"extra:{extra}"]

        X = X.reindex(columns=self.feature_names)
        latest = X.iloc[-1:].to_numpy(dtype=np.float32, copy=False)

        if not np.isfinite(latest).all():
            return None, ["non_finite"]

        return latest, None

    def _load_costs(self) -> dict:
        """Load trading cost assumptions for audit context."""
        cfg_path = Path("config/components/costs.yaml")
        defaults = {
            "commission_bps": 0.3,
            "half_spread_bps": 0.7,
            "slippage_bps_per_turnover": 0.2,
        }
        if cfg_path.exists() and yaml is not None:
            try:
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                for k, v in defaults.items():
                    data.setdefault(k, v)
                return data
            except Exception as e:
                logger.warning(f"Failed to load costs.yaml: {e}; using defaults")
        return defaults

    def add_bar(self, symbol: str, bar: dict) -> None:
        """
        Add a new bar to the history for a symbol.

        Args:
            symbol: Symbol name
            bar: Bar data with keys: timestamp, open, high, low, close, volume
        """
        required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_keys = [key for key in required_keys if key not in bar]
        if missing_keys:
            raise ValueError(f"Missing required keys in bar: {missing_keys}")

        # Convert bar to DataFrame row
        bar_df = pd.DataFrame([{
            'Open': bar['open'],
            'High': bar['high'],
            'Low': bar['low'],
            'Close': bar['close'],
            'Volume': bar['volume']
        }], index=[pd.to_datetime(bar['timestamp'])])

        # Add to history
        if symbol not in self.history:
            self.history[symbol] = bar_df
        else:
            self.history[symbol] = pd.concat([self.history[symbol], bar_df])

        logger.debug(f"Added bar for {symbol}: {bar['timestamp']}")

    def predict_one(self, symbol: str) -> float | None:
        """
        Make a prediction for a single symbol.

        Args:
            symbol: Symbol to predict

        Returns:
            Prediction value or None if insufficient history
        """
        # Kill-switch
        if os.getenv("SERVE_KILL", "0") == "1":
            logger.warning("Serve kill-switch active; skipping prediction")
            return None

        if symbol not in self.history:
            logger.warning(f"No history for symbol: {symbol}")
            return None

        df = self.history[symbol]

        if len(df) < self.min_history_bars:
            logger.warning(f"Insufficient history for {symbol}: {len(df)} < {self.min_history_bars}")
            return None

        try:
            latest_features, errors = self._prepare_features(df)
            if errors:
                logger.warning(f"Feature prep failed for {symbol}: {errors}")
                return None

            # Make prediction
            start_time = time.perf_counter()
            yhat = self._predict_fn(latest_features)
            prediction = float(yhat[0]) if isinstance(yhat, list | np.ndarray) else float(yhat)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Telemetry
            self._num_predictions += 1
            self._last_latency_ms = latency_ms

            # Log prediction
            self._log_prediction(symbol, latest_features[0], prediction, latency_ms)

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def predict_batch(self, symbols: list[str]) -> dict[str, float | None]:
        """
        Make predictions for multiple symbols.

        Args:
            symbols: List of symbols to predict

        Returns:
            Dictionary of symbol -> prediction
        """
        results = {}

        for symbol in symbols:
            results[symbol] = self.predict_one(symbol)

        return results

    def _log_prediction(self, symbol: str, features: np.ndarray, prediction: float, latency_ms: float):
        """Log prediction details"""
        # Create prediction log entry
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "symbol": symbol,
            "features_hash": hash(tuple(features)),
            "prediction": prediction,
            "latency_ms": latency_ms,
            "model_path": self.model_path,
            "costs": self.costs,
        }

        # Save to JSONL file
        log_file = Path("artifacts/predictions.jsonl")
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.debug(f"Prediction logged: {symbol} -> {prediction:.4f} ({latency_ms:.2f}ms)")

    def get_status(self) -> dict:
        """Get adapter status"""
        return {
            "model_path": self.model_path,
            "feature_names": self.feature_names,
            "min_history_bars": self.min_history_bars,
            "symbols_with_history": list(self.history.keys()),
            "history_lengths": {symbol: len(df) for symbol, df in self.history.items()},
            "onnx_enabled": bool(self._onnx_sess is not None),
            "num_predictions": self._num_predictions,
            "last_latency_ms": self._last_latency_ms,
            "costs": self.costs,
        }


def create_adapter_from_experiment(exp_id: str) -> ModelAdapter:
    """
    Create a model adapter from an experiment ID.

    Args:
        exp_id: Experiment ID

    Returns:
        ModelAdapter instance
    """
    # Find the experiment directory
    exp_dir = Path(f"artifacts/models/{exp_id}")
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Find the model file (check both exp_dir and parent directory)
    model_files = list(exp_dir.glob("*_model.json"))
    if not model_files:
        # Check parent directory
        parent_dir = exp_dir.parent
        model_files = list(parent_dir.glob(f"{exp_id}_model.json"))

    if not model_files:
        raise FileNotFoundError(f"No model file found for experiment {exp_id}")

    model_path = str(model_files[0])

    # Load experiment metadata
    exp_meta_file = exp_dir.parent / f"{exp_id}.json"
    if exp_meta_file.exists():
        with open(exp_meta_file) as f:
            metadata = json.load(f)
        feature_names = metadata.get("features", ["ret_1d_lag1", "sma_10", "sma_20", "vol_10", "rsi_14"])
    else:
        feature_names = ["ret_1d_lag1", "sma_10", "sma_20", "vol_10", "rsi_14"]

    return ModelAdapter(model_path, feature_names)


# Thin ONNX predictor with guards/telemetry
try:
    import onnxruntime as _ort  # type: ignore
except Exception:  # pragma: no cover
    _ort = None


class _Predictor:
    def __init__(self, onnx_path: str, sidecar_path: str, min_history: int = 50, kill_env: str = "SERVE_DUMMY"):
        self.kill = os.getenv(kill_env, "0") == "1"
        with open(sidecar_path) as f:
            self.meta = json.load(f)
        self.features = self.meta.get("features") or self.meta.get("feature_names")
        if not isinstance(self.features, list):
            raise ValueError("sidecar missing 'features' list")
        self.dtypes = self.meta.get("dtypes", ["float32"] * len(self.features))
        self.min_history = min_history
        if not self.kill:
            if _ort is None:
                raise RuntimeError("onnxruntime not available")
            self.sess = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore
            self.inp = self.sess.get_inputs()[0].name

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        if cols != self.features:
            raise ValueError(f"schema mismatch: {cols} vs {self.features}")
        if len(df) < self.min_history:
            raise ValueError(f"insufficient history: need ≥{self.min_history}, have {len(df)}")
        for c, dt in zip(df.columns, self.dtypes, strict=False):
            if str(df[c].dtype) != dt:
                df[c] = df[c].astype(dt)

        # Schema CRC integrity check
        import zlib
        schema_sig = "|".join([f"{c}:{str(df[c].dtype)}" for c in df.columns])
        crc = zlib.crc32(schema_sig.encode("utf-8")) & 0xffffffff
        if "schema_crc32" in self.meta and crc != self.meta["schema_crc32"]:
            raise ValueError(f"schema CRC mismatch: got {crc}, expected {self.meta['schema_crc32']}")
        return df

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        df = self._validate(df)
        if self.kill:
            return np.zeros(len(df), dtype=np.float32)
        X = np.ascontiguousarray(df.astype("float32").values)
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.inp: X})[0].reshape(-1).astype(np.float32, copy=False)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        if os.getenv("SERVE_TELEMETRY", "1") == "1":
            Path("artifacts/serve").mkdir(parents=True, exist_ok=True)
            with open("artifacts/serve/preds.jsonl", "a") as f:
                f.write(json.dumps({"n": len(out), "lat_ms": round(lat_ms, 3)}) + "\n")
        return out


def main():
    """CLI interface for testing the adapter"""
    import argparse

    parser = argparse.ArgumentParser(description="Model adapter CLI")
    parser.add_argument("--csv", help="CSV file to process (bars for ModelAdapter or features for ONNX Predictor)")
    parser.add_argument("--exp-id", help="Experiment ID to load model from")
    parser.add_argument("--symbol", default="SPY", help="Symbol to predict")
    parser.add_argument("--onnx", help="Path to ONNX model for Predictor path (optional)")
    parser.add_argument("--sidecar", default="artifacts/parity/sidecar.json", help="Sidecar with features schema for ONNX")
    parser.add_argument("--out", default="", help="Write summary JSON to this path")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode: no trades, only log predictions")
    parser.add_argument("--schema-crc", type=int, help="Expected schema CRC32 for validation")
    args = parser.parse_args()

    # If ONNX path is provided, run thin Predictor path on feature CSV
    if args.onnx:
        if not args.csv:
            print("No --csv provided for ONNX Predictor path")
            return 1
        try:
            # Lazy import to avoid hard dependency if unused
            import json as _json
            try:
                with open(args.sidecar) as f:
                    meta = _json.load(f)
                features = meta.get("features") or meta.get("feature_names")
            except Exception:
                features = None
            df = pd.read_csv(args.csv)
            # Coerce schema ordering if possible
            if features and list(df.columns) != features:
                with contextlib.suppress(Exception):
                    df = df.reindex(columns=features)
            pred = _Predictor(args.onnx, args.sidecar).predict_batch(df)
            print(f"[SERVE] batch={len(pred)} done")
            return 0
        except Exception as e:
            print(f"[SERVE] ONNX predictor failed: {e}")
            return 1

    if args.exp_id:
        adapter = create_adapter_from_experiment(args.exp_id)
    else:
        # Use the latest experiment
        exp_dirs = sorted(Path("artifacts/models").glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not exp_dirs:
            print("No experiments found")
            return 1

        latest_exp = exp_dirs[0].name
        adapter = create_adapter_from_experiment(latest_exp)

    print(f"Using model from experiment: {adapter.model_path}")
    print(f"Status: {adapter.get_status()}")

    # Shadow mode setup
    if args.shadow:
        print("[SHADOW] Running in shadow mode - predictions only, no trades")
        shadow_log = Path("artifacts/shadow/predictions.jsonl")
        shadow_log.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    preds_made = 0
    last_pred: float | None = None
    if args.csv:
        # Process CSV file
        df = pd.read_csv(args.csv)
        print(f"Processing {len(df)} bars from {args.csv}")

        for _, row in df.iterrows():
            bar = {
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            adapter.add_bar(args.symbol, bar)
            total_rows += 1

            # Make prediction after enough history
            if len(adapter.history[args.symbol]) >= adapter.min_history_bars:
                prediction = adapter.predict_one(args.symbol)
                if prediction is not None:
                    preds_made += 1
                    last_pred = float(prediction)

                    # Shadow mode logging
                    if args.shadow:
                        shadow_entry = {
                            "timestamp": row['timestamp'],
                            "symbol": args.symbol,
                            "prediction": float(prediction),
                            "latency_ms": adapter._last_latency_ms or 0.0,
                            "model_path": adapter.model_path,
                            "shadow": True
                        }
                        with open(shadow_log, "a") as f:
                            f.write(json.dumps(shadow_entry) + "\n")
                        print(f"[SHADOW] {row['timestamp']}: {args.symbol} -> {prediction:.4f}")
                    else:
                        print(f"{row['timestamp']}: {args.symbol} -> {prediction:.4f}")
    # Optional ONNX override (only if requested and available)
    # No ONNX requested; using native path

    if args.out:
        out = {
            "rows": int(total_rows),
            "predictions_made": int(preds_made),
            "last_prediction": float(last_pred) if last_pred is not None else None,
            "onnx": bool(args.onnx and (adapter._onnx_sess is not None)),
            "min_history_bars": int(adapter.min_history_bars),
            "last_latency_ms": float(adapter._last_latency_ms) if adapter._last_latency_ms is not None else None,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote summary to {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
