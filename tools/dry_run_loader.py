#!/usr/bin/env python3
"""
Enhanced dry-run data loader with proper lookback for rolling features.
Handles last trading day detection and sufficient history.
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import os
import json

# Import our calendar utilities
from tools.last_trading_day import last_trading_day, validate_trading_day

# Configuration
LOOKBACK_DAYS = 60  # Sufficient for 20-day rolling features + buffer
MIN_SYMBOLS_FOR_RANKING = 50


def load_feature_whitelist():
    """Load the protected feature whitelist."""
    whitelist_path = Path("results/production/features_whitelist.json")
    
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Feature whitelist not found: {whitelist_path}")
    
    with open(whitelist_path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'features' in data:
        return data['features']
    else:
        raise ValueError(f"Invalid whitelist format in {whitelist_path}")


def create_realistic_mock_data(symbols, start_date, end_date):
    """Create realistic mock data with sufficient lookback for rolling features."""
    
    # Generate business days only
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    
    all_bars = []
    np.random.seed(42)  # Reproducible
    
    print(f"📊 Generating mock data for {len(symbols)} symbols over {len(all_dates)} business days")
    
    for symbol in symbols:
        # Each symbol has different characteristics
        base_price = 50 + np.random.uniform(10, 200)
        volatility = np.random.uniform(0.15, 0.35)  # 15-35% annual vol
        drift = np.random.uniform(-0.05, 0.15)      # -5% to +15% annual drift
        
        price = base_price
        
        for i, date in enumerate(all_dates):
            # Random walk with realistic parameters
            daily_drift = drift / 252  # Annualized to daily
            daily_vol = volatility / np.sqrt(252)  # Annualized to daily
            
            daily_return = np.random.normal(daily_drift, daily_vol)
            price *= (1 + daily_return)
            
            # Realistic intraday OHLC
            intraday_range = abs(daily_return) + np.random.uniform(0.005, 0.025)
            
            open_offset = np.random.uniform(-0.3, 0.3) * intraday_range
            close_offset = daily_return
            
            open_price = price / (1 + daily_return) * (1 + open_offset)
            close_price = price
            
            high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.7) * intraday_range)
            low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.7) * intraday_range)
            
            # Realistic volume (higher on volatile days)
            base_volume = np.random.lognormal(12, 1.2)  # ~160k average
            vol_multiplier = 1 + 2 * abs(daily_return)  # Higher volume on big moves
            volume = int(base_volume * vol_multiplier)
            
            all_bars.append({
                'date': date.date(),
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
    
    df = pd.DataFrame(all_bars)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    print(f"✅ Generated {len(df):,} bars")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    
    return df


def load_bars_for_target(symbols, target_date, use_alpaca=True):
    """
    Load bars with sufficient lookback for rolling features.
    
    Args:
        symbols: List of symbols to fetch
        target_date: Target date for signal generation
        use_alpaca: If True, try Alpaca API; if False or fails, use mock data
        
    Returns:
        DataFrame with OHLCV data including sufficient lookback
    """
    # Calculate date range with lookback
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date).date()
    
    start_date = target_date - timedelta(days=LOOKBACK_DAYS + 10)  # Extra buffer for weekends/holidays
    end_date = target_date
    
    print(f"📅 Loading bars for target date: {target_date}")
    print(f"   Lookback range: {start_date} to {end_date}")
    print(f"   Symbols: {len(symbols)}")
    
    if use_alpaca:
        try:
            # Try to use Alpaca API
            api_key = os.environ.get('APCA_API_KEY_ID')
            secret_key = os.environ.get('APCA_API_SECRET_KEY')
            
            if api_key and secret_key:
                print("📡 Attempting to fetch from Alpaca...")
                # Import fetch function
                from tools.fetch_bars_alpaca import fetch_alpaca_bars
                
                # Use real symbols if possible, otherwise mock
                real_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
                fetch_symbols = real_symbols[:min(len(real_symbols), len(symbols))]
                
                df = fetch_alpaca_bars(fetch_symbols, start_date.strftime('%Y-%m-%d'), 
                                     end_date.strftime('%Y-%m-%d'), '1Day', 'iex')
                
                if not df.empty and len(df) > 100:  # Reasonable amount of data
                    print(f"✅ Alpaca data fetched: {len(df)} bars")
                    return df
                else:
                    print("⚠️  Insufficient Alpaca data, falling back to mock")
            else:
                print("⚠️  Alpaca credentials not found, using mock data")
                
        except Exception as e:
            print(f"⚠️  Alpaca fetch failed: {e}, using mock data")
    
    # Fallback to mock data
    print("📊 Generating realistic mock data...")
    df = create_realistic_mock_data(symbols, start_date, end_date)
    
    return df


def validate_features_for_target(df, target_date, feature_whitelist):
    """
    Validate that we have sufficient features for the target date.
    
    Args:
        df: Raw OHLCV data 
        target_date: Target date for validation
        feature_whitelist: List of required feature names
        
    Returns:
        dict with validation results
    """
    
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date).date()
    
    print(f"🔍 Validating features for {target_date}...")
    
    # Check if target date is in the data
    target_data = df[df['date'] == target_date]
    
    validation = {
        'target_date': target_date,
        'is_trading_day': validate_trading_day(target_date),
        'data_available': len(target_data) > 0,
        'symbols_available': target_data['symbol'].nunique() if len(target_data) > 0 else 0,
        'sufficient_symbols': False,
        'sufficient_lookback': False,
        'min_date_in_data': df['date'].min() if len(df) > 0 else None,
        'max_date_in_data': df['date'].max() if len(df) > 0 else None,
        'total_bars': len(df),
        'feature_whitelist_count': len(feature_whitelist)
    }
    
    # Check symbol count
    validation['sufficient_symbols'] = validation['symbols_available'] >= MIN_SYMBOLS_FOR_RANKING
    
    # Check lookback (need at least 30 days of data before target)
    min_required_date = target_date - timedelta(days=30)
    validation['sufficient_lookback'] = (
        validation['min_date_in_data'] is not None and 
        validation['min_date_in_data'] <= min_required_date
    )
    
    # Overall validation
    validation['ready_for_features'] = (
        validation['is_trading_day'] and
        validation['data_available'] and 
        validation['sufficient_symbols'] and
        validation['sufficient_lookback']
    )
    
    return validation


def main():
    """Test the dry-run loader."""
    
    print("🧪 TESTING DRY-RUN LOADER")
    print("=" * 30)
    
    # Get last trading day
    target = last_trading_day()
    print(f"📅 Target date: {target}")
    
    # Load feature whitelist
    try:
        whitelist = load_feature_whitelist()
        print(f"📋 Feature whitelist: {len(whitelist)} features")
        print(f"   Sample: {whitelist[:3]}...")
    except Exception as e:
        print(f"⚠️  Whitelist error: {e}")
        whitelist = []
    
    # Test symbols
    symbols = [f"SYM{i:03d}" for i in range(100)]  # Mock symbols
    
    # Load data
    df = load_bars_for_target(symbols, target)
    
    # Validate
    validation = validate_features_for_target(df, target, whitelist)
    
    print(f"\n🔍 VALIDATION RESULTS:")
    for key, value in validation.items():
        status = "✅" if value else "❌" if isinstance(value, bool) else "📊"
        print(f"   {key:<25} {status} {value}")
    
    if validation['ready_for_features']:
        print(f"\n✅ READY FOR FEATURE GENERATION!")
    else:
        print(f"\n❌ NOT READY - fix issues above")
    
    return 0 if validation['ready_for_features'] else 1


if __name__ == "__main__":
    exit(main())
