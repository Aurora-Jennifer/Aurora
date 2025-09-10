"""
Real-time data ingestion from Alpaca API for paper trading.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

def fetch_alpaca_bars(symbols: List[str], timeframe: str = "5Min", lookback_minutes: int = 300) -> pd.DataFrame:
    """
    Fetch real-time bars from Alpaca API.
    
    Args:
        symbols: List of symbols to fetch
        timeframe: Bar timeframe (5Min, 1Hour, 1Day)
        lookback_minutes: How far back to fetch data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        from alpaca_trade_api import REST
        
        # Initialize Alpaca API
        api = REST(
            key_id=os.environ.get('APCA_API_KEY_ID'),
            secret_key=os.environ.get('APCA_API_SECRET_KEY'),
            base_url=os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        # Calculate time range
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        
        logger.info(f"📡 Fetching {len(symbols)} symbols from Alpaca ({timeframe})")
        
        all_bars = []
        for symbol in symbols:
            try:
                bars = api.get_bars(
                    symbol, 
                    timeframe, 
                    start.isoformat(), 
                    end.isoformat(),
                    asof=None,
                    feed='iex'  # Use IEX feed for paper trading
                ).df
                
                if not bars.empty:
                    bars["symbol"] = symbol
                    bars = bars.reset_index()
                    all_bars.append(bars)
                    logger.debug(f"   ✅ {symbol}: {len(bars)} bars")
                else:
                    logger.warning(f"   ⚠️ {symbol}: No data returned")
                    
            except Exception as e:
                logger.warning(f"   ❌ {symbol}: {e}")
                continue
        
        if not all_bars:
            raise RuntimeError("No data returned from Alpaca API")
        
        # Combine all bars
        df = pd.concat(all_bars, ignore_index=True)
        
        # Normalize schema
        df.rename(columns={
            "timestamp": "date",
            "open": "open", 
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)
        
        df["date"] = pd.to_datetime(df["date"], utc=True)
        
        # Select and order columns
        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
        
        logger.info(f"✅ Fetched {len(df)} bars for {len(symbols)} symbols")
        return df
        
    except ImportError:
        logger.error("alpaca_trade_api not installed. Install with: pip install alpaca-trade-api")
        raise
    except Exception as e:
        logger.error(f"Alpaca API error: {e}")
        raise

def fetch_intraday_panel(symbols: List[str], bar_size: str = "5Min", lookback_days: int = 30, market_tz: str = "America/New_York") -> pd.DataFrame:
    """
    Fetch combined historical daily + intraday data for fresh feature computation.
    
    Args:
        symbols: List of symbols to fetch
        bar_size: Intraday bar size (5Min, 15Min, 1Hour)
        lookback_days: Days of historical data for features
        market_tz: Market timezone
        
    Returns:
        DataFrame with combined historical + intraday data
    """
    try:
        from alpaca_trade_api import REST
        import pytz
        
        # Initialize Alpaca API
        api = REST(
            key_id=os.environ.get('APCA_API_KEY_ID'),
            secret_key=os.environ.get('APCA_API_SECRET_KEY'),
            base_url=os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        # Timezone handling
        market_tz_obj = pytz.timezone(market_tz)
        now_utc = datetime.now(timezone.utc)
        now_market = now_utc.astimezone(market_tz_obj)
        
        # Calculate time ranges
        today_start = now_market.replace(hour=9, minute=30, second=0, microsecond=0)  # Market open
        lookback_start = today_start - timedelta(days=lookback_days)
        
        # Avoid current forming bar by ending 15 seconds ago
        intraday_end = now_market - timedelta(seconds=15)
        
        logger.info(f"📡 Fetching intraday panel: {len(symbols)} symbols, {bar_size} bars")
        logger.info(f"   Historical: {lookback_start.date()} to {today_start.date()}")
        logger.info(f"   Intraday: {today_start.date()} {today_start.time()} to {intraday_end.time()}")
        
        all_bars = []
        
        for symbol in symbols:
            try:
                # 1. Fetch historical daily data (for features)
                hist_bars = api.get_bars(
                    symbol,
                    "1Day",
                    lookback_start.isoformat(),
                    today_start.isoformat(),
                    asof=None,
                    feed='iex'
                ).df
                
                # 2. Fetch intraday data (today's bars)
                intraday_bars = api.get_bars(
                    symbol,
                    bar_size,
                    today_start.isoformat(),
                    intraday_end.astimezone(timezone.utc).isoformat(),
                    asof=None,
                    feed='iex'
                ).df
                
                # Combine historical + intraday
                if not hist_bars.empty and not intraday_bars.empty:
                    combined_bars = pd.concat([hist_bars, intraday_bars])
                elif not hist_bars.empty:
                    combined_bars = hist_bars
                elif not intraday_bars.empty:
                    combined_bars = intraday_bars
                else:
                    logger.warning(f"   ⚠️ {symbol}: No data returned")
                    continue
                
                # Add symbol and normalize
                combined_bars["symbol"] = symbol
                combined_bars = combined_bars.reset_index()
                combined_bars.rename(columns={"timestamp": "date"}, inplace=True)
                combined_bars["date"] = pd.to_datetime(combined_bars["date"], utc=True)
                
                # Sort by timestamp
                combined_bars = combined_bars.sort_values("date")
                all_bars.append(combined_bars)
                
                logger.debug(f"   ✅ {symbol}: {len(hist_bars)} daily + {len(intraday_bars)} intraday = {len(combined_bars)} total")
                
            except Exception as e:
                logger.warning(f"   ❌ {symbol}: {e}")
                continue
        
        if not all_bars:
            raise RuntimeError("No data returned from Alpaca API")
        
        # Combine all symbols
        df = pd.concat(all_bars, ignore_index=True)
        
        # Select and order columns
        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
        
        # Log fetch success (freshness check handled by caller)
        latest_ts = df["date"].max()
        logger.info(f"✅ Fetched {len(df)} bars for {len(symbols)} symbols (latest: {latest_ts.strftime('%H:%M:%S')})")
        return df
        
    except ImportError:
        logger.error("alpaca_trade_api not installed. Install with: pip install alpaca-trade-api")
        raise
    except Exception as e:
        logger.error(f"Alpaca API error: {e}")
        raise

def fetch_latest_prices(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch latest prices for symbols (for real-time trading).
    
    Args:
        symbols: List of symbols
        
    Returns:
        DataFrame with latest prices
    """
    try:
        from alpaca_trade_api import REST
        
        api = REST(
            key_id=os.environ.get('APCA_API_KEY_ID'),
            secret_key=os.environ.get('APCA_API_SECRET_KEY'),
            base_url=os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        # Get latest quotes
        quotes = api.get_latest_quotes(symbols)
        
        data = []
        for symbol, quote in quotes.items():
            if quote:
                data.append({
                    'symbol': symbol,
                    'date': datetime.now(timezone.utc),
                    'close': float(quote.bid_price),  # Use bid as close approximation
                    'volume': 0  # Not available in quotes
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['open'] = df['close']  # Approximate
            df['high'] = df['close']  # Approximate  
            df['low'] = df['close']   # Approximate
            
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        logger.error(f"Error fetching latest prices: {e}")
        return pd.DataFrame()

def create_fallback_data(symbols: List[str]) -> pd.DataFrame:
    """
    Create fallback mock data when Alpaca is unavailable.
    
    Args:
        symbols: List of symbols
        
    Returns:
        DataFrame with mock OHLCV data
    """
    logger.warning("Using fallback mock data - Alpaca unavailable")
    
    data = []
    base_time = datetime.now(timezone.utc)
    
    for symbol in symbols:
        # Generate realistic mock data
        base_price = 100 + hash(symbol) % 200  # Consistent base price per symbol
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        
        close = base_price * (1 + price_change)
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        
        data.append({
            'date': base_time,
            'symbol': symbol,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': int(np.random.uniform(100000, 1000000))
        })
    
    return pd.DataFrame(data)

def check_data_freshness(df: pd.DataFrame, bar_size: str = "5Min", lag_seconds: int = 90) -> bool:
    """
    Check if data is fresh enough for trading decisions.
    
    Uses proper bar boundary logic: only demand the last COMPLETED bar,
    not the bar that's still closing. Adds grace window for provider posting lag.
    
    Args:
        df: DataFrame with 'date' column
        bar_size: Expected bar size (5Min, 15Min, etc.)
        lag_seconds: Grace window for provider posting lag
        
    Returns:
        True if data is fresh, False if stale
    """
    if df.empty:
        logger.warning("STALE_DATA: Empty DataFrame")
        return False
    
    import pytz
    
    # Market timezone
    MARKET_TZ = pytz.timezone("America/New_York")
    BAR = pd.Timedelta(bar_size)
    LAG = pd.Timedelta(f"{lag_seconds}s")
    
    def now_et():
        return datetime.now(MARKET_TZ)
    
    def floor_to_5m(dt):
        return dt.replace(second=0, microsecond=0) - timedelta(minutes=dt.minute % 5)
    
    def expected_last_completed_end(et_now):
        # At 15:35:06, floor→15:35:00; last *completed* ends at 15:30:00
        return floor_to_5m(et_now) - timedelta(minutes=5)
    
    # Get latest bar timestamp (should be UTC)
    latest_ts_utc = df["date"].max()
    
    # Calculate expected last completed bar end time
    et_now = now_et()
    expected_et = expected_last_completed_end(et_now)
    expected_utc = expected_et.astimezone(pytz.UTC)
    
    # Allow late posting with grace window
    freshness_cutoff = expected_utc - LAG
    is_fresh = latest_ts_utc >= freshness_cutoff
    
    # Calculate lag for logging
    lag_actual = (expected_utc - latest_ts_utc).total_seconds()
    
    if is_fresh:
        logger.info(f"✅ FRESH_DATA: latest_utc={latest_ts_utc.strftime('%H:%M:%S')}, "
                   f"expected_utc={expected_utc.strftime('%H:%M:%S')}, lag={lag_actual:.0f}s, fresh=True")
    else:
        logger.warning(f"⚠️ STALE_DATA: latest_utc={latest_ts_utc.strftime('%H:%M:%S')}, "
                      f"expected_utc={expected_utc.strftime('%H:%M:%S')}, lag={lag_actual:.0f}s, fresh=False")
    
    return is_fresh

def compute_feature_hash(X: pd.DataFrame) -> str:
    """
    Compute a hash of the feature matrix for change detection.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Short hash string for logging
    """
    try:
        # Convert to numpy array and compute hash
        arr = X.values
        hash_obj = hashlib.blake2b(arr.tobytes(), digest_size=8)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute feature hash: {e}")
        return "hash_error"

def compute_raw_entropy(raw_predictions: np.ndarray, temperature: float = 0.1) -> float:
    """
    Compute entropy on raw predictions before normalization.
    
    Args:
        raw_predictions: Raw model predictions
        temperature: Temperature for softmax
        
    Returns:
        Entropy in nats
    """
    try:
        # Apply softmax with temperature
        exp_preds = np.exp(raw_predictions / temperature)
        probs = exp_preds / np.sum(exp_preds)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return entropy
    except Exception as e:
        logger.warning(f"Failed to compute raw entropy: {e}")
        return 0.0
