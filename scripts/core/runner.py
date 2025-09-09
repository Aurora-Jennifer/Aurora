#!/usr/bin/env python3
"""
Paper Execution Loop - E2E with broker (paper only)
Loop: fetch bars → DataSanity → features → inference → risk → broker → telemetry
"""
import argparse
import asyncio
import json
import os
import sys
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.metrics.comprehensive import create_metrics_collector
from core.guards.price import price_sane, extract_rolling_prices, has_corporate_action_today
from scripts.core.e2d import E2DRunner
from brokers.realtime_feed import RealtimeFeed, is_realtime_enabled, check_trading_halted


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
        self.last_processed_ts = None  # Prevent reprocessing same data
        self.universe_symbols = set()  # Track active universe

        # Initialize comprehensive metrics
        run_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = create_metrics_collector(run_id)
        
        # Clearframe guardrails 
        self.NOTIONAL_CAP_DAY = 2000  # $2k daily cap
        self.MAX_QTY_PER_SYMBOL = 1  # 1 share pilot mode
        self.today_notional = {}  # symbol -> daily notional
        
        # Price guard configuration
        self.price_guard_config = self.profile.get("guards", {}).get("price", {
            "jump_limit_frac": 0.30,
            "band_lookback_bars": 90,
            "band_frac": 0.80,
            "warmup_bars": 30,
            "absurd_max": 1000000
        })

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
    
    def _log_clearframe_metrics(self, symbol: str, df: pd.DataFrame, decisions: list, predict_start: float):
        """Log Clearframe's 4 key metrics to prove bar advancement."""
        predict_ms = (time.perf_counter() - predict_start) * 1000
        
        # Extract bar metrics
        last_bar_end = df.index[-1]
        last_bar_close = float(df["Close"].iloc[-1])  # Note: capitalized column name
        
        # Create feature hash (simplified - hash the last few OHLCV values)
        recent_data = df.tail(5).values.flatten()
        feat_hash = hashlib.md5(recent_data.tobytes()).hexdigest()[:8]
        
        # Extract confidence if available
        confidence = None
        if decisions:
            confidence = decisions[0].get("confidence", "N/A")
        
        print(f"[CLEARFRAME] {symbol} | bar_end={last_bar_end} | close={last_bar_close:.4f} | feat={feat_hash} | predict_ms={predict_ms:.1f} | conf={confidence}")
        
        self._log_telemetry("clearframe_metrics", {
            "symbol": symbol,
            "last_bar_end": str(last_bar_end),
            "last_bar_close": last_bar_close,
            "feature_hash": feat_hash,
            "predict_ms": predict_ms,
            "confidence": confidence
        })

    def _fetch_latest_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch latest bars - live data if live=true, else historical simulation."""
        bars = {}
        window_size = 200  # Increased window for better ML features
        
        # Check if this is a live profile
        is_live_flag = self.profile["data"].get("live", False)
        
        if is_live_flag:
            # LIVE MODE: Fetch current data from yfinance
            import yfinance as yf
            print("[FETCH] 🔴 LIVE MODE: Fetching current data from yfinance")
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    # Get live 1-minute data for last 5 days as specified in profile
                    df = ticker.history(period="5d", interval="1m", auto_adjust=True)
                    
                    if len(df) == 0:
                        print(f"[FETCH] ❌ No live data for {symbol}")
                        continue
                        
                    # Ensure timezone and standard columns
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
                    elif df.index.tz != "UTC":
                        df.index = df.index.tz_convert("UTC")
                    
                    # Take last window_size bars for ML features
                    if len(df) > window_size:
                        df = df.tail(window_size)
                    
                    bars[symbol] = df[["Open", "High", "Low", "Close", "Volume"]]
                    latest_bar = df.index[-1]
                    age_minutes = (pd.Timestamp.now(tz="UTC") - latest_bar).total_seconds() / 60
                    print(f"[FETCH] ✅ {symbol}: {len(df)} bars, latest={latest_bar}, age={age_minutes:.1f}min")
                    
                except Exception as e:
                    print(f"[FETCH] ❌ Failed to fetch live {symbol}: {e}")
                    continue
                    
            return bars

        # HISTORICAL MODE: Load from files
        for symbol in symbols:
            # First try Kaggle dataset (highest quality)
            kaggle_path = Path(f"data/training/kaggle/{symbol}.parquet")
            if kaggle_path.exists():
                df = pd.read_parquet(kaggle_path)
                historical_window = self._get_simulation_window(df, symbol, window_size)
                bars[symbol] = historical_window[["Open", "High", "Low", "Close", "Volume"]]
                continue
                
            # Try yfinance dataset (multi-regime data)
            yfinance_path = Path(f"data/training/yfinance/{symbol}.parquet")
            if yfinance_path.exists():
                df = pd.read_parquet(yfinance_path)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                historical_window = self._get_simulation_window(df, symbol, window_size)
                bars[symbol] = historical_window[["Open", "High", "Low", "Close", "Volume"]]
                continue
                
            # Fallback to fixtures (original smoke data)
            fixture_path = Path(f"data/fixtures/smoke/{symbol}.csv")
            if fixture_path.exists():
                df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                historical_window = self._get_simulation_window(df, symbol, window_size)
                bars[symbol] = historical_window[["Open", "High", "Low", "Close", "Volume"]]
                continue
                
            # Final fallback to cache
            cache_path = Path(f"data/smoke_cache/{symbol}.parquet")
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                historical_window = self._get_simulation_window(df, symbol, window_size)
                
                # Ensure OHLCV columns exist
                if "Open" not in df.columns:
                    close = df.iloc[:, 0]
                    historical_window = pd.DataFrame({
                        "Open": close,
                        "High": close * 1.002,
                        "Low": close * 0.998,
                        "Close": close,
                        "Volume": 1000000
                    }, index=historical_window.index)
                bars[symbol] = historical_window[["Open", "High", "Low", "Close", "Volume"]]


        self._log_telemetry("fetch_bars", {"symbols": symbols, "n_bars": sum(len(df) for df in bars.values())})
        return bars
    
    def _get_simulation_window(self, df: pd.DataFrame, symbol: str, window_size: int) -> pd.DataFrame:
        """Get time-advancing simulation window for historical data."""
        # Initialize simulation position tracking if not exists
        if not hasattr(self, 'simulation_positions'):
            self.simulation_positions = {}
            
        # Initialize this symbol's position if first time
        if symbol not in self.simulation_positions:
            # Start from position window_size to ensure we have enough history
            self.simulation_positions[symbol] = window_size
            
        current_pos = self.simulation_positions[symbol]
        
        # Check bounds - if we're at end of data, stay there
        if current_pos >= len(df):
            current_pos = len(df) - 1
            self.simulation_positions[symbol] = current_pos
            
        # Get window ending at current position
        start_idx = max(0, current_pos - window_size + 1)
        end_idx = current_pos + 1
        
        window = df.iloc[start_idx:end_idx].copy()
        
        # Advance position for next loop (simulate time progression)
        if current_pos < len(df) - 1:
            self.simulation_positions[symbol] += 1
            
        return window

    def _data_sanity_check(self, bars: dict[str, pd.DataFrame]) -> bool:
        """Quick DataSanity checks."""
        try:
            for symbol, df in bars.items():
                print(f"[SANITY] Checking {symbol}: {len(df)} bars, range {df.index.min()} to {df.index.max()}")

                # Basic range checks
                if df["High"].max() < df["Low"].min():
                    print(f"[SANITY] BREACH: {symbol} high_low_inversion")
                    self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "high_low_inversion"})
                    return False

                # Check for required columns
                required = ["Open", "High", "Low", "Close", "Volume"]
                missing = [col for col in required if col not in df.columns]
                if missing:
                    print(f"[SANITY] BREACH: {symbol} missing_columns: {missing}")
                    self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "missing_columns", "missing": missing})
                    return False

                # Basic value validation
                if df.isnull().any().any():
                    print(f"[SANITY] BREACH: {symbol} null_values")
                    self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "null_values"})
                    return False

                # Enhanced staleness check with bar-interval awareness
                latest_time = df.index.max()
                # Detect historical vs live data
                snapshot_path = self.profile["data"].get("snapshot", "")
                is_live_flag = self.profile["data"].get("live", False)
                is_historical = (("yfinance" in snapshot_path or "training" in snapshot_path) and not is_live_flag) 
                
                if is_historical:
                    # Historical data: allow old data for backtesting
                    max_age = timedelta(days=30*365)
                    age = pd.Timestamp.now(tz="UTC") - latest_time
                    if age > max_age:
                        print(f"[SANITY] BREACH: {symbol} stale_data, age={age}")
                        self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "stale_data", "latest": str(latest_time)})
                        return False
                else:
                    # Live data: use bar-interval-based staleness
                    if len(df) >= 2:
                        bar_sec = int((df.index[-1] - df.index[-2]).total_seconds())
                        # Check for demo relaxed staleness
                        demo_max_hours = self.profile.get("demo", {}).get("max_staleness_hours", None)
                        
                        if demo_max_hours:
                            # Demo mode: use relaxed staleness
                            tolerance_sec = demo_max_hours * 3600  # hours to seconds
                            age_sec = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds()
                            if age_sec > tolerance_sec:
                                print(f"[SANITY] BREACH: {symbol} stale_demo_data, age={age_sec/3600:.1f}h vs max={demo_max_hours}h")
                                self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "stale_demo_data", "age_hours": age_sec/3600, "max_hours": demo_max_hours})
                                return False
                            print(f"[SANITY] DEMO OK: {symbol} age={age_sec/3600:.1f}h vs max={demo_max_hours}h")
                        else:
                            # Production mode: strict bar-based tolerance
                            tolerance_sec = max(180, 3 * max(bar_sec, 1))
                            age_sec = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds()
                            if age_sec > tolerance_sec:
                                print(f"[SANITY] BREACH: {symbol} stale_live_data, age={age_sec}s vs tolerance={tolerance_sec}s")
                                self._log_telemetry("sanity_breach", {"symbol": symbol, "issue": "stale_live_data", "age_sec": age_sec, "tolerance_sec": tolerance_sec})
                                return False
                
                print(f"[SANITY] {symbol} staleness: OK, historical={is_historical}")

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
        if not decision:
            return True
            
        risk_config = self.profile["risk"]

        # Check daily loss limit
        daily_pnl = self._get_daily_pnl()
        if daily_pnl < -risk_config["daily_loss_limit"]:
            print(f"[RISK] VETO: Daily loss limit exceeded: {daily_pnl} < -{risk_config['daily_loss_limit']}")
            self._log_telemetry("risk_veto", {"reason": "daily_loss_limit", "decision": decision.get("symbol", "unknown")})
            return True

        # Check position limits
        symbol = decision.get("symbol", "")
        current_pos = self.positions.get(symbol, 0)
        if abs(current_pos) >= risk_config["max_dollar"]:
            print(f"[RISK] VETO: Position limit exceeded for {symbol}: {current_pos} >= {risk_config['max_dollar']}")
            self._log_telemetry("risk_veto", {"reason": "position_limit", "decision": symbol})
            return True

        # Check confidence threshold
        confidence = decision.get("confidence", 0)
        if confidence < 0.05:
            print(f"[RISK] VETO: Low confidence for {symbol}: {confidence} < 0.05")
            self._log_telemetry("risk_veto", {"reason": "low_confidence", "decision": symbol})
            return True

        print(f"[RISK] APPROVED: {symbol} with confidence {confidence}")
        return False

    def _get_daily_pnl(self) -> float:
        """Calculate daily PnL (mock)."""
        # In real implementation, calculate from trades
        return 0.0
    
    def _get_market_price(self, symbol: str) -> float | None:
        """Get current market price for symbol from latest data."""
        # Try Kaggle dataset first
        kaggle_path = Path(f"data/training/kaggle/{symbol}.parquet")
        if kaggle_path.exists():
            df = pd.read_parquet(kaggle_path)
            return float(df['Close'].iloc[-1])
        
        # Fallback to fixtures
        fixture_path = Path(f"data/fixtures/smoke/{symbol}.csv")
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
            return float(df['Close'].iloc[-1])
        
        # Final fallback to cache
        cache_path = Path(f"data/smoke_cache/{symbol}.parquet")
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return float(df.iloc[-1, 0])  # Assume first column is price
        
        print(f"[PRICE] WARNING: No price data found for {symbol}")
        return None

    def _execute_trade(self, decision: dict[str, Any], bars: dict[str, pd.DataFrame] = None) -> dict[str, Any]:
        """Execute trade through paper broker."""
        if self.mode == "paper":
            symbol = decision["symbol"]
            side = decision["side"]
            desired_qty = decision["qty"]
            
            # IDEMPOTENCY: Check if we're trying to repeat the same order
            current_qty = self.positions.get(symbol, 0)
            
            # For BUY orders, check if we already have enough position
            if side == "BUY" and current_qty >= desired_qty:
                print(f"[TRADE] SKIP {symbol}: already have {current_qty}, no need for more")
                return {"status": "skipped", "reason": "sufficient_position"}
            
            # For SELL orders, check if we already sold enough
            if side == "SELL" and current_qty <= -desired_qty:
                print(f"[TRADE] SKIP {symbol}: already short {current_qty}, no need for more")
                return {"status": "skipped", "reason": "sufficient_short"}
            
            # Get current simulation price from bars (preferred) or fallback to historical
            if bars and symbol in bars:
                price = float(bars[symbol]['Close'].iloc[-1])
                print(f"[PRICE] Using current simulation price for {symbol}: {price}")
            else:
                price = self._get_market_price(symbol)
                print(f"[PRICE] Using fallback price for {symbol}: {price}")
            
            if price is None or price <= 0:
                raise ValueError(f"No valid price for {symbol}")
            
            # CLEARFRAME ADAPTIVE PRICE VALIDATION
            if bars and symbol in bars:
                symbol_bars = bars[symbol]
                rolling_prices = extract_rolling_prices(symbol_bars, self.price_guard_config.get("band_lookback_bars", 90))
                
                # Get previous price for jump check (if available)
                p_tm1 = None
                if len(symbol_bars) >= 2:
                    p_tm1 = float(symbol_bars['Close'].iloc[-2])
                
                # Check for corporate actions (TODO: integrate with real CA data)
                bar_timestamp = symbol_bars.index[-1] if hasattr(symbol_bars.index, '__getitem__') else None
                had_corp_action = has_corporate_action_today(symbol, bar_timestamp) if bar_timestamp else False
                
                # Run comprehensive price sanity check
                is_sane, reason = price_sane(
                    symbol=symbol,
                    p_t=price,
                    p_tm1=p_tm1,
                    rolling_prices=rolling_prices,
                    had_corp_action_today=had_corp_action,
                    config=self.price_guard_config
                )
                
                if not is_sane:
                    self._log_telemetry("trade_rejected", {"symbol": symbol, "reason": reason, "price": price})
                    print(f"[PRICE] 🛑 {symbol} rejected: {reason} (price={price:.4f})")
                    return {"status": "rejected", "reason": f"price_guard_failed: {reason}"}
                print(f"[PRICE] ✅ {symbol} passed price guards (price={price:.4f})")
            
            # CLEARFRAME SIZING CAPS WITH DETAILED LOGGING
            
            # 0. Staleness gate: cap to zero for historical data (as Clearframe noted)
            snapshot_path = self.profile["data"].get("snapshot", "")
            is_live_flag = self.profile["data"].get("live", False)
            is_historical = (("yfinance" in snapshot_path or "training" in snapshot_path) and not is_live_flag)
            if is_historical:
                latest_time = bars[symbol].index.max()
                age_days = (pd.Timestamp.now(tz="UTC") - latest_time).days
                print(f"[RISK] CAP {symbol} qty {desired_qty}→0 reason=historical_data last_bar={latest_time} age_days={age_days}")
                self._log_telemetry("trade_rejected", {
                    "symbol": symbol, 
                    "reason": "historical_data", 
                    "requested_qty": desired_qty,
                    "capped_qty": 0,
                    "last_bar": str(latest_time),
                    "age_days": age_days
                })
                return {"status": "rejected", "reason": "qty_capped_to_zero"}
            
            # 1. Hard limit: 1 share per symbol
            current_qty = self.positions.get(symbol, 0)
            max_additional = self.MAX_QTY_PER_SYMBOL - abs(current_qty)
            
            original_qty = desired_qty
            cap_reason = None
            cap_details = {}
            
            if desired_qty > max_additional:
                capped_qty = max_additional
                cap_reason = "max_qty_per_symbol"
                cap_details = {
                    "requested_qty": original_qty,
                    "current_qty": current_qty,
                    "max_additional": max_additional,
                    "capped_qty": capped_qty
                }
                print(f"[RISK] CAP {symbol} qty {original_qty}→{capped_qty} reason={cap_reason} current={current_qty}")
                self._log_telemetry("trade_capped", {"symbol": symbol, "reason": cap_reason, **cap_details})
                desired_qty = capped_qty
            
            if desired_qty <= 0:
                final_reason = cap_reason or "zero_after_caps"
                print(f"[RISK] REJECT {symbol} qty={original_qty} reason={final_reason}")
                return {"status": "rejected", "reason": "qty_capped_to_zero"}
            
            # 2. Daily notional cap
            order_notional = desired_qty * price
            today_total = sum(self.today_notional.values())
            if today_total + order_notional > self.NOTIONAL_CAP_DAY:
                self._log_telemetry("trade_rejected", {"symbol": symbol, "reason": "notional_cap", "order_notional": order_notional, "today_total": today_total})
                return {"status": "rejected", "reason": f"notional_cap: would exceed ${self.NOTIONAL_CAP_DAY}"}
            
            # Mock execution with real price
            fill = {
                "symbol": symbol,
                "side": side,
                "qty": desired_qty,
                "price": price,
                "timestamp": pd.Timestamp.now().isoformat(),
                "fill_id": f"fill_{len(self.trades)}",
                "commission": desired_qty * 0.001,  # Mock commission
                "slippage": price * 0.0001  # 1 bps slippage
            }

            # Update positions - set absolute position, not additive
            if side == "BUY":
                self.positions[symbol] = current_qty + desired_qty
            else:
                self.positions[symbol] = current_qty - desired_qty
            
            # Track daily notional
            if symbol not in self.today_notional:
                self.today_notional[symbol] = 0
            self.today_notional[symbol] += order_notional
                
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
        
        # Set active universe for validation
        self.universe_symbols = set(symbols)

        loop_count = 0

        while time.perf_counter() < end_time:
            loop_start = time.perf_counter()

            try:
                # 1. Fetch latest bars
                bars = self._fetch_latest_bars(symbols)
                
                # GUARD: Check if we're reprocessing same data (skip for demo mode)
                if bars:
                    latest_bar_ts = max(df.index.max() for df in bars.values())
                    # Allow reprocessing for historical demo/testing
                    snapshot_path = self.profile["data"].get("snapshot", "")
                    is_live_flag = self.profile["data"].get("live", False)
                    is_historical = (("yfinance" in snapshot_path or "training" in snapshot_path) and not is_live_flag)
                    if (self.last_processed_ts and latest_bar_ts <= self.last_processed_ts and not is_historical):
                        print(f"[RUNNER] SKIP: No new data since {self.last_processed_ts}")
                        time.sleep(1)
                        continue
                    self.last_processed_ts = latest_bar_ts

                # 2. DataSanity quick checks
                if not self._data_sanity_check(bars):
                    print(f"[RUNNER] ⚠️  DataSanity breach, skipping loop {loop_count}")
                    time.sleep(1)
                    continue

                # 3. Run E2D to get decisions
                predict_start = time.perf_counter()
                decisions = self._run_e2d(bars)
                
                # 3.5. Log Clearframe metrics for each symbol
                for symbol, df in bars.items():
                    self._log_clearframe_metrics(symbol, df, decisions, predict_start)

                # 4. Execute trades
                for decision in decisions:
                    symbol = decision.get('symbol', 'UNKNOWN')
                    
                    # ASSERTION: Symbol must be in active universe
                    assert symbol in self.universe_symbols, f"FATAL: Decision for {symbol} not in universe {self.universe_symbols}"
                    
                    if not self._risk_veto(decision):
                        fill = self._execute_trade(decision, bars)
                        if isinstance(fill, dict) and 'side' in fill:
                            print(f"[RUNNER] 💰 {fill['side']} {fill['qty']} {fill['symbol']} @ {fill['price']}")
                        else:
                            print(f"[RUNNER] ⏭️  {symbol}: {fill.get('reason', 'skipped')}")
                    else:
                        print(f"[RUNNER] 🛑 Risk veto: {symbol}")

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

    async def run_realtime(self, symbol: str, minutes: int) -> dict[str, Any]:
        """Run paper trading with real-time WebSocket feed."""
        print(f"[RUNNER] 🚀 Starting realtime paper loop for {minutes} minutes")
        print(f"[RUNNER] 📊 Symbol: {symbol}")
        
        # Feature flag check
        if not is_realtime_enabled():
            print("[RUNNER] ⚠️  Realtime mode requires FLAG_REALTIME=1")
            return {"status": "disabled", "reason": "FLAG_REALTIME=0"}
        
        start_time = time.perf_counter()
        end_time = start_time + (minutes * 60)
        bar_count = 0
        
        # Initialize realtime feed
        feed = RealtimeFeed(symbol, "1m", testnet=True)
        
        # Track last processed bar timestamp
        last_bar_ts = None
        
        def on_bar(bar_data):
            """Process each new bar through E2D pipeline."""
            nonlocal bar_count, last_bar_ts
            
            try:
                # Check trading halt
                if check_trading_halted():
                    print("[RUNNER] 🛑 Trading halted, skipping bar")
                    return
                
                bar_ts = bar_data["timestamp"]
                
                # Skip duplicate timestamps  
                if last_bar_ts and bar_ts <= last_bar_ts:
                    print(f"[RUNNER] ⏭️  Skipping duplicate bar: {bar_ts}")
                    return
                    
                last_bar_ts = bar_ts
                bar_count += 1
                
                print(f"[RUNNER] 📊 Processing bar {bar_count}: {symbol} @ {bar_data['close']:.2f}")
                
                # Convert to DataFrame format for E2D
                df = pd.DataFrame([{
                    'Open': bar_data['open'],
                    'High': bar_data['high'], 
                    'Low': bar_data['low'],
                    'Close': bar_data['close'],
                    'Volume': bar_data['volume'],
                    'symbol': symbol
                }], index=[bar_ts])
                
                # Track latency  
                feed_ts = bar_data.get('feed_ts', time.time())
                model_start_ts = time.time()
                
                # Run E2D pipeline
                runner = E2DRunner(self.profile)
                result = runner.run(df)
                
                model_end_ts = time.time()
                
                # Process decisions
                if result.get("status") == "success":
                    decisions = result.get("decisions", [])
                    
                    for decision in decisions:
                        # Track decision latency
                        decision_ts = time.time()
                        
                        # Execute trade (create bars structure for price access)
                        current_bars = {symbol: df} if 'symbol' in locals() and df is not None else None
                        trade_result = self._execute_trade(decision, current_bars)
                        
                        broker_ts = time.time()
                        
                        # Log latency telemetry
                        latency_ms = {
                            'feed_to_model': (model_start_ts - feed_ts) * 1000,
                            'model_duration': (model_end_ts - model_start_ts) * 1000,  
                            'decision_to_broker': (broker_ts - decision_ts) * 1000,
                            'total_latency': (broker_ts - feed_ts) * 1000
                        }
                        
                        self._log_telemetry("realtime_trade", {
                            "bar_count": bar_count,
                            "symbol": symbol,
                            "bar_ts": bar_ts.isoformat(),
                            "latency_ms": latency_ms,
                            "trade_result": trade_result
                        })
                        
                        print(f"[RUNNER] ⚡ Latency: {latency_ms['total_latency']:.1f}ms total")
                        
                else:
                    print(f"[RUNNER] ⚠️  E2D failed: {result.get('error', 'unknown')}")
                    
            except Exception as e:
                print(f"[RUNNER] ❌ Error processing bar: {e}")
                self._log_telemetry("realtime_error", {"error": str(e), "bar_count": bar_count})
        
        def on_heartbeat(staleness_sec):
            """Monitor feed health."""
            if staleness_sec > 3:
                print(f"[RUNNER] 💓 Feed staleness: {staleness_sec:.1f}s")
        
        def on_halt(reason):
            """Handle trading halt."""
            print(f"[RUNNER] 🛑 TRADING HALTED: {reason}")
            
        # Wire callbacks
        feed.on_bar = on_bar
        feed.on_heartbeat = on_heartbeat  
        feed.on_halt = on_halt
        
        # Start feed (async)
        print("[RUNNER] 📡 Connecting to realtime feed...")
        
        try:
            # Run with timeout
            await asyncio.wait_for(feed.start(), timeout=minutes * 60)
            
        except TimeoutError:
            print(f"[RUNNER] ⏰ Realtime run completed ({minutes} minutes)")
            
        except Exception as e:
            print(f"[RUNNER] ❌ Realtime error: {e}")
            
        total_time = time.perf_counter() - start_time
        
        # Final stats
        feed_stats = feed.get_stats()
        
        print(f"[RUNNER] ✅ Realtime completed: {bar_count} bars, {len(self.trades)} trades in {total_time:.1f}s")
        
        return {
            "status": "completed",
            "mode": "realtime", 
            "bar_count": bar_count,
            "n_trades": len(self.trades),
            "total_time": total_time,
            "positions": self.positions,
            "feed_stats": feed_stats,
            "metrics_summary": self.metrics.get_current_metrics()
        }


def main():
    parser = argparse.ArgumentParser(description="Paper execution runner")
    parser.add_argument("--profile", default="config/profiles/golden_xgb_v2.yaml", help="Profile path")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"], help="Execution mode")
    parser.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    parser.add_argument("--minutes", type=int, default=15, help="Run duration in minutes")
    parser.add_argument("--source", default="csv", choices=["csv", "feed", "realtime"], help="Data source (csv=stub, feed=live, realtime=websocket)")
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

    # Initialize runner
    runner = PaperRunner(args.profile, args.mode, args.source, args.feed_url)
    
    # Run based on source mode
    if args.source == "realtime":
        # Realtime mode (async) - single symbol only for now
        if len(symbols) > 1:
            print("[RUNNER] ⚠️  Realtime mode supports single symbol only, using first: " + symbols[0])
        
        result = asyncio.run(runner.run_realtime(symbols[0], args.minutes))
    else:
        # Static mode (csv/feed)
        result = runner.run(symbols, args.minutes)

    # Save artifacts
    runner._save_artifacts(args.out)

    trades_count = result.get('n_trades', 0)
    print(f"[RUNNER] 🎉 Paper loop complete: {trades_count} trades executed")
    return 0


if __name__ == "__main__":
    exit(main())
