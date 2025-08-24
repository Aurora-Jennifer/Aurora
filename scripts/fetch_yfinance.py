#!/usr/bin/env python3
"""
AURORA Data Acquisition - Rung 2: Multi-Regime Equity Data
Fetch OHLCV data via yfinance with corporate actions and regime classification.
"""

import os
import sys
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from datetime import datetime, timezone
import argparse
from typing import List, Dict, Any
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.data_sanity import DataSanityValidator


# Regime periods for classification
REGIMES = {
    "dot_com_crash": ("2000-03-01", "2002-10-01"),
    "financial_crisis": ("2007-10-01", "2009-03-01"), 
    "covid_crash": ("2020-02-01", "2020-04-01"),
    "rate_hike_selloff": ("2022-01-01", "2022-10-01"),
    "steady_growth": ("2010-01-01", "2019-12-31"),
    "post_covid_bull": ("2020-05-01", "2021-12-31")
}

# Default symbol universe - broad market representation
DEFAULT_SYMBOLS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "TLT",   # 20+ Year Treasury
    "GLD",   # Gold
    "VXX",   # Volatility
    "AAPL",  # Mega cap tech
    "MSFT",  # Mega cap tech
    "GOOGL", # Mega cap tech
    "BRK-B", # Value/diversified
    "JNJ",   # Defensive
    "XOM",   # Energy/commodities
]


def setup_logging():
    """Configure logging for data acquisition."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def classify_regime(date: str) -> str:
    """Classify a date into a market regime."""
    date_obj = pd.to_datetime(date).date()
    
    for regime, (start, end) in REGIMES.items():
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        if start_date <= date_obj <= end_date:
            return regime
    
    return "normal"


def fetch_symbol_data(symbol: str, start_date: str, end_date: str, logger) -> pd.DataFrame:
    """Fetch OHLCV data for a single symbol."""
    try:
        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,  # Keep raw prices + separate adjustment data
            prepost=False,
            repair=True
        )
        
        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Standardize column names to match system expectations
        data = data.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Ensure UTC timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data.index = data.index.tz_convert('UTC')
            
        # Add regime classification
        data['regime'] = data.index.strftime('%Y-%m-%d').map(classify_regime)
        
        logger.info(f"✅ {symbol}: {len(data)} bars, {data.index[0].date()} to {data.index[-1].date()}")
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def clean_and_validate(data: pd.DataFrame, symbol: str, logger) -> pd.DataFrame:
    """Apply standard cleaning pipeline."""
    if data.empty:
        return data
        
    logger.info(f"Cleaning {symbol} data...")
    
    # 1. Cast OHLCV to float64
    price_cols = ['Open', 'High', 'Low', 'Close']
    data[price_cols] = data[price_cols].astype('float64')
    data['Volume'] = data['Volume'].astype('float64')
    
    # 2. Remove duplicates
    initial_len = len(data)
    data = data[~data.index.duplicated(keep='first')]
    if len(data) < initial_len:
        logger.warning(f"Removed {initial_len - len(data)} duplicate timestamps")
    
    # 3. Sort by timestamp (ensure monotonic)
    data = data.sort_index()
    
    # 4. Basic sanity checks
    if not data.index.is_monotonic_increasing:
        logger.error(f"Non-monotonic timestamps for {symbol}")
        
    # Check for negative prices
    negative_prices = (data[price_cols] <= 0).any(axis=1).sum()
    if negative_prices > 0:
        logger.warning(f"Found {negative_prices} bars with negative/zero prices")
        data = data[(data[price_cols] > 0).all(axis=1)]
    
    # Check OHLC consistency
    ohlc_invalid = (
        (data['High'] < data['Low']) |
        (data['High'] < data['Open']) |
        (data['High'] < data['Close']) |
        (data['Low'] > data['Open']) |
        (data['Low'] > data['Close'])
    ).sum()
    
    if ohlc_invalid > 0:
        logger.warning(f"Found {ohlc_invalid} bars with invalid OHLC relationships")
        # Filter out invalid bars
        valid_ohlc = (
            (data['High'] >= data['Low']) &
            (data['High'] >= data['Open']) &
            (data['High'] >= data['Close']) &
            (data['Low'] <= data['Open']) &
            (data['Low'] <= data['Close'])
        )
        data = data[valid_ohlc]
    
    logger.info(f"✅ Cleaned {symbol}: {len(data)} valid bars")
    return data


def run_datasanity_validation(data: pd.DataFrame, symbol: str, logger) -> bool:
    """Run DataSanity validation if available."""
    try:
        validator = DataSanityValidator(profile="default")
        
        # Use validate_and_repair which returns (clean_data, validation_result)
        validated_data, result = validator.validate_and_repair(data, symbol=symbol)
        
        if result.repairs:
            logger.info(f"✅ DataSanity validation passed for {symbol} with {len(result.repairs)} repairs")
        else:
            logger.info(f"✅ DataSanity validation passed for {symbol} (no repairs needed)")
        return True
            
    except Exception as e:
        logger.warning(f"DataSanity validation failed for {symbol}: {e}")
        return False  # Report validation failures but don't block


def save_data(data: pd.DataFrame, symbol: str, output_dir: Path, logger):
    """Save cleaned data to parquet format."""
    if data.empty:
        logger.warning(f"No data to save for {symbol}")
        return
        
    output_file = output_dir / f"{symbol}.parquet"
    data.to_parquet(output_file, compression='snappy')
    logger.info(f"💾 Saved {symbol}: {output_file} ({len(data)} bars)")


def generate_manifest(symbols: List[str], output_dir: Path, logger):
    """Generate manifest file with dataset metadata."""
    manifest_data = []
    
    for symbol in symbols:
        parquet_file = output_dir / f"{symbol}.parquet"
        if not parquet_file.exists():
            continue
            
        try:
            data = pd.read_parquet(parquet_file)
            
            regime_counts = data['regime'].value_counts().to_dict()
            
            manifest_entry = {
                'symbol': symbol,
                'file': str(parquet_file.relative_to(output_dir.parent)),
                'start_date': data.index.min().isoformat(),
                'end_date': data.index.max().isoformat(),
                'bar_count': len(data),
                'avg_volume': float(data['Volume'].mean()),
                'regime_coverage': regime_counts,
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }
            manifest_data.append(manifest_entry)
            
        except Exception as e:
            logger.error(f"Failed to process {symbol} for manifest: {e}")
    
    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    logger.info(f"📋 Generated manifest: {manifest_file}")
    
    # Also save as CSV for easy inspection
    if manifest_data:
        df = pd.DataFrame(manifest_data)
        csv_file = output_dir / "manifest.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"📋 Generated CSV manifest: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch multi-regime equity data via yfinance")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                      help="Symbols to fetch (default: broad market universe)")
    parser.add_argument("--start", default="2000-01-01", 
                      help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01",
                      help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/training/yfinance",
                      help="Output directory for parquet files")
    parser.add_argument("--skip-validation", action="store_true",
                      help="Skip DataSanity validation")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting yfinance data acquisition")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Period: {args.start} to {args.end}")
    logger.info(f"   Output: {output_dir}")
    
    success_count = 0
    
    for symbol in args.symbols:
        try:
            # Fetch raw data
            data = fetch_symbol_data(symbol, args.start, args.end, logger)
            if data.empty:
                continue
            
            # Clean and validate
            data = clean_and_validate(data, symbol, logger)
            if data.empty:
                continue
                
            # Run DataSanity validation
            if not args.skip_validation:
                validation_passed = run_datasanity_validation(data, symbol, logger)
                if not validation_passed:
                    logger.warning(f"DataSanity validation failed for {symbol}, but continuing...")
            
            # Save to parquet
            save_data(data, symbol, output_dir, logger)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            continue
    
    # Generate manifest
    generate_manifest(args.symbols, output_dir, logger)
    
    logger.info(f"🎉 Completed: {success_count}/{len(args.symbols)} symbols successful")
    
    if success_count > 0:
        logger.info(f"✅ Data ready for training at: {output_dir}")
        logger.info(f"   Next: Update config/profiles to point to new dataset")
        logger.info(f"   Next: Retrain models with expanded regime coverage")


if __name__ == "__main__":
    main()
