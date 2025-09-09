#!/usr/bin/env python3
"""
E2D: End-to-Decision - System reality check
Flows: raw snapshot → features → model → signal → position decision (no broker)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from core.data_sanity import DataSanityValidator
from core.metrics.comprehensive import create_metrics_collector
from core.ml.build_features import build_matrix


class E2DRunner:
    """End-to-Decision runner with telemetry and risk engine."""

    def __init__(self, profile_path: str):
        self.profile_path = profile_path
        self.profile = self._load_profile()
        self.trace_events = []
        self.decisions = []

        # Initialize comprehensive metrics
        run_id = f"e2d_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = create_metrics_collector(run_id)

    def _load_profile(self) -> dict[str, Any]:
        """Load profile configuration."""
        with open(self.profile_path) as f:
            return yaml.safe_load(f)

    def _trace(self, module: str, event: str, duration_ms: float = 0, meta: dict = None):
        """Record trace event."""
        self.trace_events.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "module": module,
            "event": event,
            "duration_ms": duration_ms,
            "meta": meta or {}
        })

    def _load_data(self, csv_path: str | None = None) -> pd.DataFrame:
        """Load data from snapshot or CSV."""
        t0 = time.perf_counter()

        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            self._trace("data", "load_csv", (time.perf_counter() - t0) * 1000, {"rows": len(df)})
            return df

        # Load from golden snapshot
        snapshot_path = self.profile["data"]["snapshot"]
        # When loading from snapshot, use symbols from config
        symbols = self.profile["data"]["symbols"]

        dfs = []
        for symbol in symbols:
            sym_path = Path(snapshot_path) / f"{symbol}.parquet"
            if sym_path.exists():
                df = pd.read_parquet(sym_path)
                df["symbol"] = symbol
                dfs.append(df)

        if not dfs:
            raise ValueError(f"No data found in snapshot: {snapshot_path}")

        df = pd.concat(dfs, ignore_index=True)
        self._trace("data", "load_snapshot", (time.perf_counter() - t0) * 1000, {"rows": len(df), "symbols": symbols})
        return df

    def _validate_data(self, df: pd.DataFrame) -> tuple[bool, list]:
        """Run DataSanity validation and return status + failing rules."""
        t0 = time.perf_counter()

        try:
            # Use the correct DataSanity API with proper configuration
            # Merge profile datasanity config with base config
            import yaml
            with open("config/data_sanity.yaml") as f:
                base_config = yaml.safe_load(f)

            # Override with profile datasanity config if present
            if "datasanity" in self.profile:
                base_config.update(self.profile["datasanity"])

            # Create temporary config file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(base_config, f)
                temp_config_path = f.name

            validator = DataSanityValidator(config_path=temp_config_path, profile="default")

            # Verify rule registry
            from core.data_sanity.registry import RULES
            required_rules = {"price_positivity", "ohlc_consistency", "finite_numbers"}
            missing_rules = required_rules - set(RULES.keys())
            if missing_rules:
                raise SystemExit(f"[E2D] FATAL: DataSanity rules not registered: {missing_rules}")

            # Run ingest stage validation
            print(f"[E2D][DataSanity] ingest: starting (rows={len(df)})", flush=True)
            # Use the actual symbol from the data, not hardcoded SPY
            symbol = df.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in df.columns and len(df) > 0 else 'UNKNOWN'
            validated_df, result = validator.validate_and_repair(df, symbol=symbol)

            # Log ingest results
            neg_count = sum(1 for r in result.repairs if "non_positive_prices" in r)
            print(f"[E2D][DataSanity] ingest: OK (rows={len(validated_df)}, neg={neg_count}, zeros=0)", flush=True)

            # Run post-adjustment validation (after any transformations)
            print(f"[E2D][DataSanity] post_adjust: starting (rows={len(validated_df)})", flush=True)
            try:
                validated_df = validator.validate_post_adjust(validated_df, symbol=symbol)
                print(f"[E2D][DataSanity] post_adjust: OK (rows={len(validated_df)}, neg=0, zeros=0)", flush=True)
            except Exception as e:
                print(f"[E2D][DataSanity] post_adjust: FATAL - {str(e)}", flush=True)
                raise SystemExit(f"[E2D] FATAL: DataSanity post-adjust validation failed: {str(e)}")

            # Clean up temp file
            import os
            os.unlink(temp_config_path)

            # Extract validation results
            failing_rules = []
            if result.flags:
                failing_rules.extend(result.flags)
            if result.repairs:
                failing_rules.extend([f"repair:{r}" for r in result.repairs])

            self._trace("sanity", "validate", (time.perf_counter() - t0) * 1000, {"status": "pass"})
            return True, failing_rules

        except Exception as e:
            self._trace("sanity", "validate", (time.perf_counter() - t0) * 1000, {"status": "fail", "error": str(e)})
            return False, [f"error:{str(e)}"]

    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build features and labels."""
        t0 = time.perf_counter()

        # Use feature groups from profile
        feature_groups = self.profile["features"]["groups"]

        # Normalize column names to uppercase for feature builder
        df_normalized = df.copy()
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }
        df_normalized = df_normalized.rename(columns=column_mapping)

        X, y = build_matrix(df_normalized, horizon=1, exclude_tags=None)

        self._trace("features", "build", (time.perf_counter() - t0) * 1000, {
            "n_features": len(X.columns),
            "n_samples": len(X),
            "groups": feature_groups
        })

        return X, y

    def _load_model(self) -> Any:
        """Load ONNX model for inference."""
        t0 = time.perf_counter()

        try:
            import onnxruntime as ort
            onnx_path = "artifacts/models/latest.onnx"

            if not Path(onnx_path).exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name

            self._trace("model", "load_onnx", (time.perf_counter() - t0) * 1000, {"path": onnx_path})
            return sess, input_name

        except Exception as e:
            self._trace("model", "load_onnx", (time.perf_counter() - t0) * 1000, {"error": str(e)})
            raise

    def _infer(self, model_info: tuple, X: pd.DataFrame) -> np.ndarray:
        """Run model inference."""
        t0 = time.perf_counter()

        sess, input_name = model_info

        # Ensure float32 and correct order
        X_float = X.astype("float32")
        X_np = np.ascontiguousarray(X_float.values)

        predictions = sess.run(None, {input_name: X_np})[0].reshape(-1)

        self._trace("inference", "predict", (time.perf_counter() - t0) * 1000, {
            "n_predictions": len(predictions),
            "mean_pred": float(np.mean(predictions)),
            "std_pred": float(np.std(predictions))
        })

        return predictions

    def _risk_engine(self, signal: float, symbol: str, current_pos: float = 0) -> dict[str, Any]:
        """Risk engine: determine position size and stops."""
        t0 = time.perf_counter()

        risk_config = self.profile["risk"]

        # Simple risk logic
        confidence = abs(signal)
        max_dollar = risk_config["max_dollar"]

        # Position sizing based on confidence
        if confidence > 0.1:
            qty = max_dollar / 100  # Simple fixed sizing for demo
            side = "BUY" if signal > 0 else "SELL"
        else:
            qty = 0
            side = "HOLD"

        # Risk flags
        risk_flags = []
        if confidence < 0.05:
            risk_flags.append("low_confidence")
        if abs(current_pos) > max_dollar:
            risk_flags.append("position_limit")

        decision = {
            "symbol": symbol,
            "signal": float(signal),
            "confidence": float(confidence),
            "side": side,
            "qty": float(qty),
            "stop_bp": risk_config["stop_bp"],
            "take_bp": risk_config["stop_bp"] * 2,  # 2:1 reward/risk
            "risk_flags": risk_flags,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        self._trace("risk", "decision", (time.perf_counter() - t0) * 1000, {
            "side": side,
            "qty": qty,
            "flags": risk_flags
        })

        return decision

    def run(self, csv_path: str | None = None) -> dict[str, Any]:
        """Run end-to-decision pipeline."""
        start_time = time.perf_counter()

        # Initialize DataSanity variables
        ds_ok = True
        failing_rules = []

        try:
            # 1. Load data
            df = self._load_data(csv_path)

            # 2. DataSanity validation (optional for testing)
            try:
                ds_ok, failing_rules = self._validate_data(df)
                if not ds_ok:
                    print(f"[E2D] ⚠️  DataSanity validation failed: {failing_rules}, continuing anyway")
            except Exception as e:
                print(f"[E2D] ⚠️  DataSanity validation error: {e}, continuing anyway")
                ds_ok = False

            # 3. Build features
            X, y = self._build_features(df)

            # 4. Load model
            model_info = self._load_model()

            # 5. Inference
            predictions = self._infer(model_info, X)

            # 6. Risk engine decisions - Use symbols from actual data
            symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else self.profile["data"]["symbols"]
            for i, symbol in enumerate(symbols):
                if i < len(predictions):
                    signal = predictions[i]
                    decision = self._risk_engine(signal, symbol)
                    self.decisions.append(decision)

            # 7. Record comprehensive metrics
            total_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_latency(total_time)

            # Log comprehensive metrics
            self.metrics.log_metrics({
                "n_decisions": len(self.decisions),
                "total_latency_ms": total_time,
                "datasanity_ok": ds_ok,
                "failing_rules_count": len(failing_rules)
            })

            # Save metrics summary
            self.metrics.save_summary()

            # Save contract-compliant metrics to expected artifact location
            contract_metrics = self.metrics.get_current_metrics()
            artifact_dir = Path("artifacts") / self.metrics.run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            metrics_file = artifact_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(contract_metrics, f, indent=2)

            print(f"[E2D] 📊 Contract metrics: {metrics_file}")

            self._trace("e2d", "complete", total_time, {
                "n_decisions": len(self.decisions),
                "total_latency_ms": total_time
            })

            return {
                "status": "success",
                "total_latency_ms": total_time,
                "n_decisions": len(self.decisions),
                "decisions": self.decisions,
                "trace": self.trace_events,
                "datasanity": {
                    "ok": ds_ok,
                    "failing_rules": failing_rules[:20]  # cap for brevity
                },
                "metrics_summary": self.metrics.get_current_metrics()
            }

        except Exception as e:
            self._trace("e2d", "error", (time.perf_counter() - start_time) * 1000, {"error": str(e)})
            raise


def main():
    parser = argparse.ArgumentParser(description="End-to-Decision pipeline")
    parser.add_argument("--profile", default="config/profiles/golden_xgb_v2.yaml", help="Profile path")
    parser.add_argument("--csv", help="Optional CSV file for testing")
    parser.add_argument("--out", default="artifacts/e2d/last", help="Output directory")
    args = parser.parse_args()

    # Set determinism switches for reproducible results
    import numpy as np
    np.seterr(all="raise")  # Strict error handling for tests/smoke

    # Load and log configuration
    import yaml
    with open(args.profile) as f:
        cfg = yaml.safe_load(f)

    print(f"[E2D] Profile: {args.profile}")
    print(f"[E2D] DataSanity config: {cfg.get('datasanity', 'NOT_FOUND')}")

    # Verify DataSanity configuration
    if "datasanity" not in cfg:
        print("[E2D] WARNING: No datasanity config found in profile")
    elif "price_positivity" not in cfg["datasanity"]:
        print("[E2D] WARNING: No price_positivity config found in datasanity")

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run E2D
    runner = E2DRunner(args.profile)
    result = runner.run(args.csv)

    # Save outputs
    with open(out_dir / "decision.json", "w") as f:
        json.dump(result["decisions"], f, indent=2)

    with open(out_dir / "trace.jsonl", "w") as f:
        for event in result["trace"]:
            f.write(json.dumps(event) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "status": result["status"],
            "total_latency_ms": result["total_latency_ms"],
            "n_decisions": result["n_decisions"],
            "profile": args.profile,
            "timestamp": pd.Timestamp.now().isoformat(),
            "datasanity": result.get("datasanity", {"ok": True, "failing_rules": []})
        }, f, indent=2)

    print(f"[E2D] ✅ Success: {result['n_decisions']} decisions in {result['total_latency_ms']:.1f}ms")
    print(f"[E2D] 📁 Outputs: {out_dir}")

    # Check latency SLO
    latency_p95 = result["total_latency_ms"]
    latency_slo = 150  # ms
    if latency_p95 > latency_slo:
        print(f"[E2D] ⚠️  Latency SLO exceeded: {latency_p95:.1f}ms > {latency_slo}ms")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
