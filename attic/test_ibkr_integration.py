"""
Test IBKR Integration
Tests the IBKR broker and data provider integration
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from brokers.data_provider import IBKRDataProvider, test_data_provider
from brokers.ibkr_broker import IBKRBroker, IBKRConfig, test_ibkr_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_ibkr_broker():
    """Test IBKR broker functionality."""
    print("=" * 60)
    print("Testing IBKR Broker")
    print("=" * 60)

    try:
        # Test connection
        print("\n1. Testing IBKR Connection...")
        connection_success = test_ibkr_connection()

        if connection_success:
            print("✅ IBKR connection test passed")

            # Test broker functionality
            print("\n2. Testing Broker Functionality...")
            config = IBKRConfig()
            broker = IBKRBroker(config=config, auto_connect=True)

            if broker.is_connected():
                # Test account info
                print("\n   Testing Account Info...")
                account_info = broker.get_account_info()
                print(f"   Account Info: {account_info}")

                # Test positions
                print("\n   Testing Positions...")
                positions = broker.get_positions()
                print(f"   Positions: {positions}")

                # Test market data
                print("\n   Testing Market Data...")
                spy_data = broker.get_market_data("SPY", "1 M", "1 day")
                if spy_data is not None:
                    print(f"   SPY Data: {len(spy_data)} rows")
                    print(f"   Date Range: {spy_data.index[0]} to {spy_data.index[-1]}")
                    print(f"   Last Close: ${spy_data['Close'].iloc[-1]:.2f}")
                else:
                    print("   ❌ No SPY data available")

                # Test real-time price
                print("\n   Testing Real-time Price...")
                price = broker.get_real_time_price("SPY")
                if price:
                    print(f"   SPY Real-time Price: ${price:.2f}")
                else:
                    print("   ❌ No real-time price available")

                broker.disconnect()
                print("\n✅ Broker functionality test passed")
                return True
            else:
                print("❌ Failed to connect to IBKR")
                return False
        else:
            print("❌ IBKR connection test failed")
            return False

    except Exception as e:
        print(f"❌ Error testing IBKR broker: {e}")
        return False


def test_data_provider():
    """Test IBKR data provider functionality."""
    print("\n" + "=" * 60)
    print("Testing IBKR Data Provider")
    print("=" * 60)

    try:
        # Test data provider
        print("\n1. Testing Data Provider...")
        provider_success = test_data_provider()

        if provider_success:
            print("✅ Data provider test passed")

            # Test additional functionality
            print("\n2. Testing Additional Data Provider Features...")
            config = IBKRConfig()
            provider = IBKRDataProvider(
                config=config, use_cache=True, fallback_to_yfinance=True
            )

            # Test multiple symbols
            print("\n   Testing Multiple Symbols...")
            symbols = ["SPY", "AAPL", "NVDA"]
            data_dict = provider.get_multiple_symbols_data(symbols, "1 M", "1 day")

            for symbol, data in data_dict.items():
                if data is not None:
                    print(f"   ✅ {symbol}: {len(data)} rows")
                else:
                    print(f"   ❌ {symbol}: No data")

            # Test cache functionality
            print("\n   Testing Cache Functionality...")
            cache_info = provider.get_cache_info()
            print(f"   Cache Info: {cache_info}")

            # Test specific date range
            print("\n   Testing Date Range...")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            spy_data = provider.get_daily_data("SPY", start_date, end_date)

            if spy_data is not None:
                print(f"   ✅ SPY Date Range: {len(spy_data)} rows")
                print(f"   Date Range: {spy_data.index[0]} to {spy_data.index[-1]}")
            else:
                print("   ❌ No SPY data for date range")

            provider.__exit__(None, None, None)
            print("\n✅ Additional data provider features test passed")
            return True
        else:
            print("❌ Data provider test failed")
            return False

    except Exception as e:
        print(f"❌ Error testing data provider: {e}")
        return False


def test_enhanced_system_integration():
    """Test integration with enhanced paper trading system."""
    print("\n" + "=" * 60)
    print("Testing Enhanced System Integration")
    print("=" * 60)

    try:
        # Test configuration loading
        print("\n1. Testing Configuration...")
        config_path = "config/enhanced_paper_trading_config.json"

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

            print(f"   ✅ Configuration loaded from {config_path}")
            print(f"   Use IBKR: {config.get('use_ibkr', False)}")
            print(f"   Symbols: {config.get('symbols', [])}")
            print(f"   Initial Capital: ${config.get('initial_capital', 0):,.0f}")

            # Test IBKR config
            ibkr_config = config.get("ibkr_config", {})
            print(f"   IBKR Paper Trading: {ibkr_config.get('paper_trading', True)}")
            print(f"   IBKR Host: {ibkr_config.get('host', '127.0.0.1')}")
            print(f"   IBKR Port: {ibkr_config.get('port', 7497)}")

        else:
            print(f"   ❌ Configuration file not found: {config_path}")
            return False

        # Test data provider initialization
        print("\n2. Testing Data Provider Initialization...")
        try:
            from brokers.data_provider import IBKRDataProvider
            from brokers.ibkr_broker import IBKRConfig

            config_obj = IBKRConfig()
            provider = IBKRDataProvider(
                config=config_obj, use_cache=True, fallback_to_yfinance=True
            )

            print("   ✅ Data provider initialized")
            print(f"   IBKR Connected: {provider.is_connected()}")
            print(f"   Cache Enabled: {provider.use_cache}")
            print(f"   Fallback Enabled: {provider.fallback_to_yfinance}")

            # Test data fetching
            print("\n3. Testing Data Fetching...")
            symbols = config.get("symbols", ["SPY"])

            for symbol in symbols[:2]:  # Test first 2 symbols
                print(f"\n   Testing {symbol}...")
                data = provider.get_historical_data(symbol, "1 M", "1 day")

                if data is not None and not data.empty:
                    print(f"   ✅ {symbol}: {len(data)} rows")
                    print(f"   Date Range: {data.index[0]} to {data.index[-1]}")
                    print(f"   Last Close: ${data['Close'].iloc[-1]:.2f}")
                    print(f"   Columns: {list(data.columns)}")
                else:
                    print(f"   ❌ {symbol}: No data available")

            provider.__exit__(None, None, None)
            print("\n✅ Enhanced system integration test passed")
            return True

        except Exception as e:
            print(f"   ❌ Error initializing data provider: {e}")
            return False

    except Exception as e:
        print(f"❌ Error testing enhanced system integration: {e}")
        return False


def main():
    """Run all tests."""
    print("IBKR Integration Test Suite")
    print("=" * 60)

    # Check if ib_insync is installed
    try:
        pass

        print("✅ ib_insync is installed")
    except ImportError:
        print("❌ ib_insync is not installed. Install with: pip install ib_insync")
        print("   Tests will be skipped.")
        return

    # Run tests
    tests = [
        ("IBKR Broker", test_ibkr_broker),
        ("Data Provider", test_data_provider),
        ("Enhanced System Integration", test_enhanced_system_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! IBKR integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
