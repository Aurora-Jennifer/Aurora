#!/usr/bin/env python3
"""
Live Trading Workflow - Research → Training → Paper Trading
==========================================================

Simple interface for live data research, model training, and paper trading.

Usage:
    python scripts/live_trading_workflow.py research --symbols SPY TSLA
    python scripts/live_trading_workflow.py train --symbols SPY --model ridge
    python scripts/live_trading_workflow.py paper --symbols SPY --duration 30
    python scripts/live_trading_workflow.py full --symbols SPY --duration 60
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import yaml
import yfinance as yf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ml.build_features import build_matrix
from ml.runtime import build_features, infer_weights
from scripts.core.paper_runner import main as paper_main


def setup_logging():
    """Setup basic logging."""
    from core.utils import setup_logging as core_setup_logging
    return core_setup_logging("logs/live_trading_workflow.log", logging.INFO)


def download_live_data(symbols: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
    """Download live data for symbols."""
    logger = setup_logging()
    data = {}
    
    for symbol in symbols:
        logger.info(f"📥 Downloading {symbol} data ({days} days)...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", auto_adjust=False)
            
            if len(df) < 50:
                logger.warning(f"⚠️  {symbol}: Only {len(df)} days of data")
                continue
                
            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
                
            data[symbol] = df
            logger.info(f"✅ {symbol}: {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol}: {e}")
    
    return data


def research_features(symbols: List[str], output_dir: str = "reports/research"):
    """Research feature performance on live data."""
    logger = setup_logging()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("🔬 Starting feature research...")
    data = download_live_data(symbols)
    
    if not data:
        logger.error("❌ No data downloaded for research")
        return
    
    results = {}
    
    for symbol, df in data.items():
        logger.info(f"📊 Analyzing {symbol}...")
        
        # Build features and calculate forward returns
        X, y = build_matrix(df, horizon=1)
        
        if len(X) < 100:
            logger.warning(f"⚠️  {symbol}: Insufficient data for analysis ({len(X)} samples)")
            continue
        
        # Calculate IC for each feature
        feature_ic = {}
        for col in X.columns:
            ic = X[col].corr(y)
            feature_ic[col] = ic
        
        # Sort by absolute IC
        sorted_features = sorted(feature_ic.items(), key=lambda x: abs(x[1]), reverse=True)
        
        results[symbol] = {
            "n_samples": len(X),
            "features": feature_ic,
            "top_features": sorted_features[:5],
            "avg_ic": sum(abs(ic) for ic in feature_ic.values()) / len(feature_ic)
        }
        
        logger.info(f"✅ {symbol}: {len(X)} samples, avg IC: {results[symbol]['avg_ic']:.4f}")
        logger.info(f"   Top features: {[f[0] for f in sorted_features[:3]]}")
    
    # Save research results
    research_file = Path(output_dir) / f"feature_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(research_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"💾 Research saved to {research_file}")
    return results


def train_live_model(symbols: List[str], model_type: str = "ridge", output_dir: str = "models"):
    """Train model on live data."""
    logger = setup_logging()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎯 Training {model_type} model on live data...")
    data = download_live_data(symbols, days=252)
    
    if not data:
        logger.error("❌ No data for training")
        return
    
    # Combine data from all symbols
    all_features = []
    all_targets = []
    
    for symbol, df in data.items():
        logger.info(f"📊 Processing {symbol}...")
        X, y = build_matrix(df, horizon=1)
        
        if len(X) < 50:
            continue
            
        all_features.append(X)
        all_targets.append(y)
    
    if not all_features:
        logger.error("❌ No valid features for training")
        return
    
    # Combine all data
    X_combined = pd.concat(all_features, ignore_index=True)
    y_combined = pd.concat(all_targets, ignore_index=True)
    
    # Remove any remaining NaNs
    mask = X_combined.notna().all(axis=1) & y_combined.notna()
    X_combined = X_combined[mask]
    y_combined = y_combined[mask]
    
    logger.info(f"📈 Training on {len(X_combined)} samples with {len(X_combined.columns)} features")
    
    # Train model
    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    else:
        logger.error(f"❌ Unknown model type: {model_type}")
        return
    
    model.fit(X_combined, y_combined)
    
    # Save model
    model_file = Path(output_dir) / f"live_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    import pickle
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature order
    feature_order = list(X_combined.columns)
    meta_file = model_file.with_suffix('.json')
    metadata = {
        "model_type": model_type,
        "feature_order": feature_order,
        "n_features": len(feature_order),
        "n_samples": len(X_combined),
        "trained_at": datetime.now().isoformat(),
        "symbols": symbols
    }
    
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model saved to {model_file}")
    logger.info(f"📋 Metadata saved to {meta_file}")
    
    return str(model_file), metadata


def paper_trade_live(symbols: List[str], duration_minutes: int = 30, model_path: str = None):
    """Run paper trading with live data."""
    logger = setup_logging()
    
    logger.info(f"🚀 Starting live paper trading for {duration_minutes} minutes...")
    logger.info(f"📈 Symbols: {symbols}")
    
    if model_path and Path(model_path).exists():
        logger.info(f"🤖 Loading model: {model_path}")
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        meta_file = Path(model_path).with_suffix('.json')
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            feature_order = metadata['feature_order']
        else:
            logger.warning("⚠️  No metadata found, using default feature order")
            feature_order = ["ret_1d_lag1", "sma_10", "sma_20", "vol_10", "rsi_14", 
                           "momentum_3d", "momentum_5d", "momentum_10d", "momentum_20d", "momentum_strength"]
    else:
        logger.info("⚠️  No model provided, using HOLD strategy")
        model = None
        feature_order = []
    
    # Initialize tracking
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    trades = []
    positions = {symbol: 0.0 for symbol in symbols}
    
    logger.info("⏰ Starting trading loop...")
    
    while datetime.now() < end_time:
        current_time = datetime.now()
        
        try:
            # Download latest data
            data = download_live_data(symbols, days=30)  # Recent data only
            
            for symbol in symbols:
                if symbol not in data:
                    continue
                
                df = data[symbol]
                if len(df) < 20:
                    continue
                
                # Build features
                F = build_features(df, feature_order)
                if len(F) == 0:
                    continue
                
                # Get latest features
                latest_features = F.iloc[-1:].values
                
                if model is not None:
                    # Predict weight
                    try:
                        weight = model.predict(latest_features)[0]
                        weight = max(-0.5, min(0.5, weight))  # Clip to ±50%
                    except Exception as e:
                        logger.warning(f"⚠️  Prediction failed for {symbol}: {e}")
                        weight = 0.0
                else:
                    weight = 0.0  # HOLD
                
                # Simple position sizing (1% per weight unit)
                position = weight * 0.01
                
                # Track if position changed
                if abs(position - positions[symbol]) > 0.001:
                    trade = {
                        "timestamp": current_time.isoformat(),
                        "symbol": symbol,
                        "weight": weight,
                        "position": position,
                        "action": "BUY" if position > positions[symbol] else "SELL"
                    }
                    trades.append(trade)
                    positions[symbol] = position
                    
                    logger.info(f"📊 {symbol}: weight={weight:.4f}, position={position:.4f}")
            
            # Wait before next update
            time.sleep(30)  # 30 second intervals
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {e}")
            time.sleep(60)  # Wait longer on error
    
    # Save results
    results = {
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "symbols": symbols,
        "trades": trades,
        "final_positions": positions,
        "model_used": model_path
    }
    
    results_file = Path("reports") / f"live_paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Paper trading completed")
    logger.info(f"📊 Total trades: {len(trades)}")
    logger.info(f"💾 Results saved to {results_file}")
    
    return results


def full_workflow(symbols: List[str], duration_minutes: int = 60):
    """Complete workflow: research → train → paper trade."""
    logger = setup_logging()
    
    logger.info("🔄 Starting full live trading workflow...")
    
    # Step 1: Research
    logger.info("=" * 50)
    logger.info("STEP 1: Feature Research")
    logger.info("=" * 50)
    research_results = research_features(symbols)
    
    # Step 2: Training
    logger.info("=" * 50)
    logger.info("STEP 2: Model Training")
    logger.info("=" * 50)
    model_path, metadata = train_live_model(symbols, model_type="ridge")
    
    # Step 3: Paper Trading
    logger.info("=" * 50)
    logger.info("STEP 3: Live Paper Trading")
    logger.info("=" * 50)
    trading_results = paper_trade_live(symbols, duration_minutes, model_path)
    
    logger.info("=" * 50)
    logger.info("🎉 Full workflow completed!")
    logger.info("=" * 50)
    
    return {
        "research": research_results,
        "training": {"model_path": model_path, "metadata": metadata},
        "trading": trading_results
    }


def main():
    parser = argparse.ArgumentParser(description="Live Trading Workflow")
    parser.add_argument("mode", choices=["research", "train", "paper", "full"], 
                       help="Workflow mode")
    parser.add_argument("--symbols", nargs="+", default=["SPY"], 
                       help="Trading symbols")
    parser.add_argument("--model", default="ridge", choices=["ridge", "xgboost"],
                       help="Model type for training")
    parser.add_argument("--duration", type=int, default=30,
                       help="Paper trading duration in minutes")
    parser.add_argument("--model-path", help="Path to existing model for paper trading")
    
    args = parser.parse_args()
    
    if args.mode == "research":
        research_features(args.symbols)
    elif args.mode == "train":
        train_live_model(args.symbols, args.model)
    elif args.mode == "paper":
        paper_trade_live(args.symbols, args.duration, args.model_path)
    elif args.mode == "full":
        full_workflow(args.symbols, args.duration)


if __name__ == "__main__":
    main()
