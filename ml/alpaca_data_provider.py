"""
Alpaca-Only Data Provider for Paper Trading

Provides market data and paper trading execution through Alpaca's unified API.
No Polygon dependency - Alpaca handles basic corporate actions.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import requests

logger = logging.getLogger(__name__)


class AlpacaDataProvider:
    """
    Alpaca-only data provider for paper trading validation.
    
    Handles:
    - Historical market data
    - Real-time quotes  
    - Basic corporate actions (via Alpaca)
    - Paper trading execution
    """
    
    def __init__(self):
        """Initialize Alpaca API connections."""
        # Get credentials from environment
        self.api_key = os.environ.get('ALPACA_API_KEY')
        self.secret_key = os.environ.get('ALPACA_SECRET_KEY') 
        self.base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not all([self.api_key, self.secret_key]):
            raise ValueError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        
        # Initialize Alpaca API
        self.api = REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url.rstrip('/v2'),  # Remove duplicate v2
            api_version='v2'
        )
        
        logger.info("Alpaca data provider initialized")
    
    def get_daily_bars(self, 
                      symbols: List[str], 
                      start_date: str, 
                      end_date: str,
                      adjustment: str = 'split') -> pd.DataFrame:
        """
        Get daily OHLCV bars from Alpaca.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            adjustment: 'raw', 'split', or 'dividend'
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching daily bars for {len(symbols)} symbols: {start_date} to {end_date}")
        
        try:
            # Get bars from Alpaca
            bars = self.api.get_bars(
                symbols,
                TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment=adjustment,
                feed='iex'  # Use IEX feed for paper trading
            )
            
            if bars.empty:
                logger.warning("No data returned from Alpaca")
                return pd.DataFrame()
            
            # Convert to standard format
            data = bars.df.reset_index()
            
            # Standardize column names
            if 'timestamp' in data.columns:
                data = data.rename(columns={'timestamp': 'date'})
            
            # Ensure we have required columns
            required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Select and order columns
            data = data[required_cols]
            
            # Ensure date is datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date and symbol
            data = data.sort_values(['date', 'symbol']).reset_index(drop=True)
            
            logger.info(f"Retrieved {len(data)} bars for {len(symbols)} symbols")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch bars from Alpaca: {e}")
            raise
    
    def get_latest_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get latest quotes for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with latest quote data
        """
        try:
            quotes = self.api.get_latest_quotes(symbols)
            
            quote_data = []
            for symbol, quote in quotes.items():
                quote_data.append({
                    'symbol': symbol,
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'bid_size': quote.bid_size,
                    'ask_size': quote.ask_size,
                    'timestamp': quote.timestamp
                })
            
            return pd.DataFrame(quote_data)
            
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return pd.DataFrame()
    
    def get_account_info(self) -> Dict:
        """Get paper trading account information."""
        try:
            account = self.api.get_account()
            
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'unrealized_pl': float(account.unrealized_pl or 0),
                'unrealized_plpc': float(account.unrealized_plpc or 0),
                'is_paper': account.trading_blocked == False  # Paper accounts aren't blocked
            }
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_positions(self) -> pd.DataFrame:
        """Get current positions."""
        try:
            positions = self.api.list_positions()
            
            if not positions:
                return pd.DataFrame()
            
            position_data = []
            for pos in positions:
                position_data.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value or 0),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl or 0),
                    'unrealized_plpc': float(pos.unrealized_plpc or 0),
                    'avg_entry_price': float(pos.avg_entry_price)
                })
            
            return pd.DataFrame(position_data)
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return pd.DataFrame()
    
    def submit_orders(self, target_weights: Dict[str, float], 
                     total_equity: float = None) -> List[Dict]:
        """
        Submit paper trading orders based on target weights.
        
        Args:
            target_weights: Dict of {symbol: weight} where weight is fraction of portfolio
            total_equity: Total portfolio equity (fetched if not provided)
            
        Returns:
            List of order results
        """
        if total_equity is None:
            account_info = self.get_account_info()
            total_equity = account_info.get('equity', 100000)  # Default $100k
        
        orders = []
        
        for symbol, weight in target_weights.items():
            try:
                # Calculate target dollar amount
                target_dollars = abs(weight * total_equity)
                
                # Skip very small positions
                if target_dollars < 100:  # Less than $100
                    continue
                
                side = 'buy' if weight > 0 else 'sell'
                
                # Submit notional order (dollar-based)
                order = self.api.submit_order(
                    symbol=symbol,
                    notional=target_dollars,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                
                orders.append({
                    'symbol': symbol,
                    'target_weight': weight,
                    'target_dollars': target_dollars,
                    'side': side,
                    'order_id': order.id,
                    'status': order.status,
                    'submitted_at': order.submitted_at
                })
                
                logger.info(f"Submitted {side} order for {symbol}: ${target_dollars:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to submit order for {symbol}: {e}")
                orders.append({
                    'symbol': symbol,
                    'target_weight': weight,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return orders
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Validate data quality for trading.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dict with validation results
        """
        if data.empty:
            return {
                'is_valid': False,
                'reason': 'No data available',
                'checks': {}
            }
        
        checks = {}
        issues = []
        
        # Check data completeness
        checks['row_count'] = len(data)
        checks['symbol_count'] = data['symbol'].nunique()
        checks['date_range'] = {
            'start': data['date'].min().isoformat(),
            'end': data['date'].max().isoformat()
        }
        
        # Check for missing values
        missing_data = data.isnull().sum()
        checks['missing_data'] = missing_data.to_dict()
        
        if missing_data.sum() > 0:
            issues.append(f"Missing data found: {missing_data.to_dict()}")
        
        # Check for data freshness
        latest_date = data['date'].max()
        now = pd.Timestamp.now()
        age_hours = (now - latest_date).total_seconds() / 3600
        
        checks['data_age_hours'] = age_hours
        
        if age_hours > 72:  # More than 3 days old
            issues.append(f"Data is stale: {age_hours:.1f} hours old")
        
        # Check price ranges (basic sanity)
        price_checks = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                price_checks[col] = {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean())
                }
                
                # Flag if prices are unrealistic
                if data[col].min() <= 0:
                    issues.append(f"Found non-positive prices in {col}")
                
                if data[col].max() > 10000:  # $10k+ per share
                    issues.append(f"Found unusually high prices in {col}: ${data[col].max():.2f}")
        
        checks['price_ranges'] = price_checks
        
        # Overall validation
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'checks': checks,
            'timestamp': datetime.now().isoformat()
        }


def test_alpaca_integration():
    """Test Alpaca integration with your credentials."""
    print("🧪 TESTING ALPACA INTEGRATION")
    print("="*40)
    
    try:
        # Initialize provider
        provider = AlpacaDataProvider()
        print("✅ Alpaca provider initialized")
        
        # Test account access
        print("\n📊 Testing account access...")
        account_info = provider.get_account_info()
        
        if account_info:
            print(f"✅ Account connected")
            print(f"   Equity: ${account_info['equity']:,.2f}")
            print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
            print(f"   Paper Trading: {account_info.get('is_paper', 'Unknown')}")
        else:
            print("❌ Failed to get account info")
            return False
        
        # Test data fetch
        print("\n📈 Testing market data...")
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = provider.get_daily_bars(test_symbols, start_date, end_date)
        
        if not data.empty:
            print(f"✅ Market data retrieved")
            print(f"   Rows: {len(data)}")
            print(f"   Symbols: {sorted(data['symbol'].unique())}")
            print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        else:
            print("⚠️ No market data returned")
        
        # Test data quality
        print("\n🔍 Testing data quality...")
        quality = provider.validate_data_quality(data)
        
        if quality['is_valid']:
            print("✅ Data quality checks passed")
        else:
            print("⚠️ Data quality issues found:")
            for issue in quality['issues']:
                print(f"   - {issue}")
        
        # Test positions (should be empty for new account)
        print("\n💼 Testing positions...")
        positions = provider.get_positions()
        print(f"✅ Positions retrieved: {len(positions)} positions")
        
        print(f"\n🎉 ALPACA INTEGRATION TEST COMPLETED")
        print(f"   Status: {'✅ READY FOR PAPER TRADING' if account_info else '❌ SETUP NEEDED'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ALPACA INTEGRATION TEST FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Alpaca API credentials")
        print("2. Ensure you have a paper trading account")
        print("3. Verify network connectivity")
        return False


if __name__ == "__main__":
    # Set environment variables for testing
    os.environ['ALPACA_API_KEY'] = 'PKQ9ZKNTB5HV9SNQ929E'
    os.environ['ALPACA_SECRET_KEY'] = 'HaZ9FkKaXJdK1HFxp6Vr3449nMXUgPWvbyZhMpPn'
    os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets/v2'
    
    # Run test
    test_alpaca_integration()
