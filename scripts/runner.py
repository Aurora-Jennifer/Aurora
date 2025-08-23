#!/usr/bin/env python3
"""
Paper Execution Loop - E2E with broker (paper only)
Loop: fetch bars → DataSanity → features → inference → risk → broker → telemetry
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.e2d import E2DRunner
from core.metrics.comprehensive import create_metrics_collector


class PaperRunner:
    """Paper trading runner with full execution loop."""

    def __init__(self, profile_path: str, mode: str = "paper", source: str = "csv", feed_url: str = None):
        self.profile_path = profile_path
        self.profile = self._load_profile()
        self.mode = mode
        self.source = source
        self.feed_url = feed_url
        self.positions = {}  # symbol -> qty
        self.trades = []
        self.telemetry = []
        
        # Initialize comprehensive metrics
        run_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = create_metrics_collector(run_id)

    def _load_profile(self) -> dict[str, Any]:
        """Load profile configuration."""
        with open(self.profile_path) as f:
            return yaml.safe_load(f)

    def _log_telemetry(self, event: str, data: dict[str, Any]):
        """Log telemetry event."""
        self.telemetry.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "event": event,
            **data
        })

    def _fetch_latest_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch latest bars for symbols (mock for now)."""
        bars = {}

        for symbol in symbols:
            # Mock bar data - in real implementation, fetch from data source
            bar_data = {
                "timestamp": pd.Timestamp.now(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000000
            }
            bars[symbol] = pd.DataFrame([bar_data])

        self._log_telemetry("fetch_bars", {"symbols": symbols, "n_bars": len(symbols)})
        return bars

    def _data_sanity_check(self, bars: dict[str, pd.DataFrame]) -> bool:
        """Quick DataSanity checks."""
        try:
            for symbol, df in bars.items():
                # Basic range checks
                if df["high"].max() < df["low"].min():
                    self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "high_low_inversion"})
                    return False

                # Staleness check (mock)
                latest_time = df["timestamp"].max()
                if pd.Timestamp.now() - latest_time > timedelta(minutes=5):
                    self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "stale_data"})
                    return False

            self._log_telemetry("sanity_check", {"status": "pass"})
            return True

        except Exception as e:
            self._log_telemetry("sanity_breach", {"issue": "exception", "error": str(e)})
            return False

    def _run_e2d(self, bars: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Run E2D pipeline to get decisions."""
        # Convert bars to CSV for E2D
        all_bars = []
        for symbol, df in bars.items():
            df_copy = df.copy()
            df_copy["symbol"] = symbol
            all_bars.append(df_copy)

        if not all_bars:
            return []

        combined_df = pd.concat(all_bars, ignore_index=True)
        temp_csv = "temp_bars.csv"
        combined_df.to_csv(temp_csv, index=False)

        try:
            # Run E2D
            runner = E2DRunner(self.profile_path)
            result = runner.run(temp_csv)

            self._log_telemetry("e2d_complete", {
                "latency_ms": result["total_latency_ms"],
                "n_decisions": len(result["decisions"])
            })

            return result["decisions"]

        finally:
            # Cleanup
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

    def _risk_veto(self, decision: dict[str, Any]) -> bool:
        """Apply risk veto rules."""
        risk_config = self.profile["risk"]

        # Check daily loss limit
        if self._get_daily_pnl() < -risk_config["daily_loss_limit"]:
            self._log_telemetry("risk_veto", {"reason": "daily_loss_limit", "decision": decision["symbol"]})
            return True

        # Check position limits
        current_pos = self.positions.get(decision["symbol"], 0)
        if abs(current_pos) >= risk_config["max_dollar"]:
            self._log_telemetry("risk_veto", {"reason": "position_limit", "decision": decision["symbol"]})
            return True

        # Check confidence threshold
        if decision["confidence"] < 0.05:
            self._log_telemetry("risk_veto", {"reason": "low_confidence", "decision": decision["symbol"]})
            return True

        return False

    def _get_daily_pnl(self) -> float:
        """Calculate daily PnL (mock)."""
        # In real implementation, calculate from trades
        return 0.0

    def _execute_trade(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Execute trade through paper broker."""
        if self.mode == "paper":
            # Mock execution
            fill = {
                "symbol": decision["symbol"],
                "side": decision["side"],
                "qty": decision["qty"],
                "price": 100.0,  # Mock price
                "timestamp": pd.Timestamp.now().isoformat(),
                "fill_id": f"fill_{len(self.trades)}",
                "commission": decision["qty"] * 0.001,  # Mock commission
                "slippage": 0.0  # Mock slippage
            }

            # Update positions
            if decision["side"] == "BUY":
                self.positions[decision["symbol"]] = self.positions.get(decision["symbol"], 0) + decision["qty"]
            elif decision["side"] == "SELL":
                self.positions[decision["symbol"]] = self.positions.get(decision["symbol"], 0) - decision["qty"]

            self.trades.append(fill)
            self._log_telemetry("trade_executed", fill)

            return fill
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _save_artifacts(self, out_dir: str):
        """Save execution artifacts."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save trades
        trades_file = out_path / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(trades_file, "w") as f:
            for trade in self.trades:
                f.write(json.dumps(trade) + "\n")

        # Save telemetry
        telemetry_file = out_path / "telemetry.jsonl"
        with open(telemetry_file, "w") as f:
            for event in self.telemetry:
                f.write(json.dumps(event) + "\n")

        # Save summary
        summary = {
            "mode": self.mode,
            "profile": self.profile_path,
            "start_time": self.telemetry[0]["timestamp"] if self.telemetry else None,
            "end_time": self.telemetry[-1]["timestamp"] if self.telemetry else None,
            "n_trades": len(self.trades),
            "n_telemetry_events": len(self.telemetry),
            "positions": self.positions,
            "daily_pnl": self._get_daily_pnl()
        }

        summary_file = out_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[RUNNER] 📁 Artifacts saved: {out_path}")

    def run(self, symbols: list[str], minutes: int = 15) -> dict[str, Any]:
        """Run paper execution loop."""
        start_time = time.perf_counter()
        end_time = start_time + (minutes * 60)

        print(f"[RUNNER] 🚀 Starting {self.mode} loop for {minutes} minutes")
        print(f"[RUNNER] 📊 Symbols: {symbols}")

        loop_count = 0

        while time.perf_counter() < end_time:
            loop_start = time.perf_counter()

            try:
                # 1. Fetch latest bars
                bars = self._fetch_latest_bars(symbols)

                # 2. DataSanity quick checks
                if not self._data_sanity_check(bars):
                    print(f"[RUNNER] ⚠️  DataSanity breach, skipping loop {loop_count}")
                    time.sleep(1)
                    continue

                # 3. Run E2D to get decisions
                decisions = self._run_e2d(bars)

                # 4. Execute trades
                for decision in decisions:
                    if not self._risk_veto(decision):
                        fill = self._execute_trade(decision)
                        print(f"[RUNNER] 💰 {fill['side']} {fill['qty']} {fill['symbol']} @ {fill['price']}")
                    else:
                        print(f"[RUNNER] 🛑 Risk veto: {decision['symbol']}")

                # 5. Record comprehensive metrics
                loop_time = (time.perf_counter() - loop_start) * 1000
                self.metrics.record_latency(loop_time)
                
                # Log telemetry
                self._log_telemetry("heartbeat", {
                    "loop_count": loop_count,
                    "latency_ms": loop_time,
                    "positions": self.positions.copy()
                })

                if loop_count % 10 == 0:
                    # Log comprehensive metrics every 10 loops
                    self.metrics.log_metrics({
                        "loop_count": loop_count,
                        "n_trades": len(self.trades),
                        "positions": self.positions.copy()
                    })
                    print(f"[RUNNER] 💓 Loop {loop_count}: {loop_time:.1f}ms, {len(self.trades)} trades")

                loop_count += 1

                # Sleep between loops
                time.sleep(1)

            except Exception as e:
                self._log_telemetry("error", {"error": str(e), "loop_count": loop_count})
                print(f"[RUNNER] ❌ Error in loop {loop_count}: {e}")
                time.sleep(1)

        total_time = time.perf_counter() - start_time
        
        # Save comprehensive metrics summary
        self.metrics.save_summary()
        
        print(f"[RUNNER] ✅ Completed: {loop_count} loops, {len(self.trades)} trades in {total_time:.1f}s")

        return {
            "status": "completed",
            "loop_count": loop_count,
            "n_trades": len(self.trades),
            "total_time": total_time,
            "positions": self.positions,
            "metrics_summary": self.metrics.get_current_metrics()
        }


def main():
    parser = argparse.ArgumentParser(description="Paper execution runner")
    parser.add_argument("--profile", default="config/profiles/golden_xgb_v2.yaml", help="Profile path")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"], help="Execution mode")
    parser.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    parser.add_argument("--minutes", type=int, default=15, help="Run duration in minutes")
    parser.add_argument("--source", default="csv", choices=["csv", "feed"], help="Data source (csv=stub, feed=live)")
    parser.add_argument("--feed-url", help="WebSocket/REST feed URL for live data")
    parser.add_argument("--out", default="artifacts/trades", help="Output directory")
    args = parser.parse_args()

    # Safety checks
    if args.mode == "live":
        if not os.getenv("ALLOW_LIVE"):
            print("[RUNNER] ❌ Live mode requires ALLOW_LIVE=true")
            return 1
        if not os.getenv("I_UNDERSTAND_LIVE_RISK"):
            print("[RUNNER] ❌ Live mode requires I_UNDERSTAND_LIVE_RISK=true")
            return 1

    symbols = [s.strip() for s in args.symbols.split(",")]

    # Run paper loop
    runner = PaperRunner(args.profile, args.mode, args.source, args.feed_url)
    result = runner.run(symbols, args.minutes)

    # Save artifacts
    runner._save_artifacts(args.out)

    print(f"[RUNNER] 🎉 Paper loop complete: {result['n_trades']} trades executed")
    return 0


if __name__ == "__main__":
    exit(main())
