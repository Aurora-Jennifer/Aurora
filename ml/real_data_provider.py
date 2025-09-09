"""
Real Data Provider Integration

Replace mock data with actual market data for paper trading validation.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class RealDataProvider:
    """
    Real market data provider using yfinance (or other APIs).
    """
    
    def __init__(self, provider: str = "yfinance"):
        """
        Initialize data provider.
        
        Args:
            provider: Data provider type ("yfinance", "alpha_vantage", "polygon")
        """
        self.provider = provider
        self.cache = {}
        logger.info(f"Initialized {provider} data provider")
    
    def fetch_daily_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        auto_adjust: bool = False) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            auto_adjust: Whether to auto-adjust for splits/dividends
            
        Returns:
            DataFrame with columns [date, symbol, open, high, low, close, volume]
        """
        logger.info(f"Fetching daily data for {len(symbols)} symbols: {start_date} to {end_date}")
        
        if self.provider == "yfinance":
            return self._fetch_yfinance_data(symbols, start_date, end_date, auto_adjust)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
    
    def _fetch_yfinance_data(self, 
                           symbols: List[str], 
                           start_date: str, 
                           end_date: str,
                           auto_adjust: bool) -> pd.DataFrame:
        """Fetch data using yfinance."""
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Download data for symbol
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    prepost=False,
                    actions=False
                )
                
                if len(data) == 0:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Reset index to get date as column
                data = data.reset_index()
                data['symbol'] = symbol
                
                # Standardize column names
                data = data.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Select required columns
                data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                all_data.append(data)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched for any symbols")
        
        # Combine all symbols
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Ensure date is datetime
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        
        # Sort by date and symbol
        combined_data = combined_data.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        logger.info(f"Fetched {len(combined_data)} rows for {len(symbols)} symbols")
        
        return combined_data
    
    def get_latest_prices(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get latest available prices for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with latest price data
        """
        # Get last 5 days to ensure we have recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        data = self.fetch_daily_data(symbols, start_date, end_date)
        
        # Get most recent date for each symbol
        latest_data = data.groupby('symbol').last().reset_index()
        
        return latest_data
    
    def validate_data_freshness(self, 
                               data: pd.DataFrame, 
                               max_age_hours: int = 24) -> Dict[str, bool]:
        """
        Validate that data is fresh enough for trading.
        
        Args:
            data: DataFrame with date column
            max_age_hours: Maximum age in hours
            
        Returns:
            Dict with validation results
        """
        if len(data) == 0:
            return {'is_fresh': False, 'reason': 'No data available'}
        
        # Get most recent date in data
        latest_date = data['date'].max()
        
        # Check age (handle timezone awareness)
        now = pd.Timestamp.now()
        if latest_date.tz is not None:
            now = now.tz_localize('UTC').tz_convert(latest_date.tz)
        age_hours = (now - latest_date).total_seconds() / 3600
        
        is_fresh = age_hours <= max_age_hours
        
        return {
            'is_fresh': is_fresh,
            'latest_date': latest_date.strftime('%Y-%m-%d %H:%M:%S'),
            'age_hours': age_hours,
            'max_age_hours': max_age_hours
        }


def create_mock_universe() -> List[str]:
    """Create a realistic stock universe for testing."""
    # S&P 500 top holdings (real symbols)
    sp500_top = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 
        'TSLA', 'GOOG', 'BRK-B', 'UNH', 'META',
        'JNJ', 'XOM', 'V', 'PG', 'JPM',
        'HD', 'CVX', 'MA', 'PFE', 'ABBV',
        'BAC', 'AVGO', 'KO', 'COST', 'DIS',
        'TMO', 'WMT', 'MRK', 'PEP', 'NFLX',
        'ABT', 'ADBE', 'VZ', 'CRM', 'DHR',
        'NKE', 'ACN', 'TXN', 'LIN', 'NEE',
        'RTX', 'CMCSA', 'QCOM', 'PM', 'HON',
        'IBM', 'T', 'SPGI', 'LOW', 'UNP'
    ]
    
    return sp500_top


def test_real_data_integration():
    """Test script to validate real data integration."""
    print("🧪 TESTING REAL DATA INTEGRATION")
    print("="*40)
    
    # Create provider
    provider = RealDataProvider("yfinance")
    
    # Test with small universe
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    try:
        # Test data fetch
        print(f"\n📊 Fetching data for {len(test_symbols)} symbols...")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = provider.fetch_daily_data(test_symbols, start_date, end_date)
        
        print(f"✅ Fetched {len(data)} rows")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   Symbols: {sorted(data['symbol'].unique())}")
        
        # Test data freshness
        print(f"\n📅 Checking data freshness...")
        freshness = provider.validate_data_freshness(data)
        
        if freshness['is_fresh']:
            print(f"✅ Data is fresh")
        else:
            print(f"⚠️ Data may be stale")
        
        print(f"   Latest: {freshness['latest_date']}")
        print(f"   Age: {freshness['age_hours']:.1f} hours")
        
        # Sample data
        print(f"\n📈 Sample data:")
        print(data.head())
        
        print(f"\n✅ REAL DATA INTEGRATION TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ REAL DATA INTEGRATION TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run test
    test_real_data_integration()
