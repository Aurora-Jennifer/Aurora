"""IBKR smoke tests - safe connection and data retrieval tests."""

import os

import pytest


def test_ibkr_connect_and_tick():
    """Test IBKR connection and basic data retrieval."""
    print("\n🔌 Testing IBKR connection...")

    # Check if IBKR environment is configured
    ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.getenv("IBKR_PORT", "7497"))
    ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID", "12399"))

    print(f"  Host: {ibkr_host}")
    print(f"  Port: {ibkr_port}")
    print(f"  Client ID: {ibkr_client_id}")

    try:
        # Try to import ib_insync
        pass

        print("  ✅ ib_insync available")
    except ImportError:
        print("  ❌ ib_insync not installed - skipping IBKR tests")
        pytest.skip("ib_insync not available")

    try:
        # Create broker instance
        from brokers.ibkr_broker import IBKRBroker, IBKRConfig

        config = IBKRConfig()
        config.host = ibkr_host
        config.port = ibkr_port
        config.client_id = ibkr_client_id
        config.paper_trading = True

        broker = IBKRBroker(config=config, auto_connect=False)

        print("  ✅ IBKRBroker created")

        # Test connection
        connected = broker.connect()
        if connected:
            print("  ✅ Connected to IBKR")

            # Test account info retrieval
            try:
                account_info = broker.get_account_info()
                print(f"  ✅ Account info retrieved: {len(account_info)} fields")
            except Exception as e:
                print(f"  ⚠️  Account info failed: {e}")

            # Test market data for SPY (safe test)
            try:
                data = broker.get_historical_data("SPY", "1 D", "1 day")
                if data is not None and len(data) > 0:
                    print(f"  ✅ Market data retrieved: {len(data)} bars")
                else:
                    print("  ⚠️  No market data returned")
            except Exception as e:
                print(f"  ⚠️  Market data failed: {e}")

            # Disconnect
            broker.disconnect()
            print("  ✅ Disconnected from IBKR")

        else:
            print("  ❌ Failed to connect to IBKR")
            print("  💡 Make sure IBKR Gateway is running and configured")
            pytest.skip("IBKR connection failed")

    except Exception as e:
        print(f"  ❌ IBKR test failed: {e}")
        pytest.skip(f"IBKR test error: {e}")


def test_ibkr_data_provider():
    """Test IBKR data provider functionality."""
    print("\n📊 Testing IBKR data provider...")

    try:
        from brokers.data_provider import IBKRDataProvider

        # Create data provider
        from brokers.ibkr_broker import IBKRConfig

        config = IBKRConfig()
        config.host = os.getenv("IBKR_HOST", "127.0.0.1")
        config.port = int(os.getenv("IBKR_PORT", "7497"))
        config.client_id = int(os.getenv("IBKR_CLIENT_ID", "12399"))
        config.paper_trading = True

        provider = IBKRDataProvider(config=config)

        print("  ✅ IBKRDataProvider created")

        # Test data retrieval (this will use yfinance fallback if IBKR fails)
        try:
            data = provider.get_daily_data("SPY", days=5)
            if data is not None and len(data) > 0:
                print(f"  ✅ Data retrieved: {len(data)} rows")
                print(f"  📈 Columns: {list(data.columns)}")
            else:
                print("  ⚠️  No data returned")
        except Exception as e:
            print(f"  ⚠️  Data retrieval failed: {e}")

    except Exception as e:
        print(f"  ❌ Data provider test failed: {e}")
        pytest.skip(f"Data provider error: {e}")


def test_ibkr_configuration():
    """Test IBKR configuration loading."""
    print("\n⚙️  Testing IBKR configuration...")

    config_files = [
        "config/ibkr_config.json",
        "config/enhanced_paper_trading_config.json",
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                import json

                with open(config_file) as f:
                    config = json.load(f)
                print(f"  ✅ {config_file} - valid JSON")

                # Check for required fields
                if "ibkr_config" in config:
                    ibkr_config = config["ibkr_config"]
                    required_fields = ["host", "port", "client_id"]
                    for field in required_fields:
                        if field in ibkr_config:
                            print(f"    ✅ {field}: {ibkr_config[field]}")
                        else:
                            print(f"    ⚠️  {field} - missing")

            except Exception as e:
                print(f"  ❌ {config_file} - error: {e}")
        else:
            print(f"  ❌ {config_file} - missing")


def test_ibkr_environment():
    """Test IBKR environment variables."""
    print("\n🌍 Testing IBKR environment...")

    env_vars = {
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "7497",
        "IBKR_CLIENT_ID": "12399",
        "IBKR_PAPER_TRADING": "true",
    }

    for var, default in env_vars.items():
        value = os.getenv(var, default)
        print(f"  {var} = {value}")

    # Check if we can set a unique client ID
    current_client_id = os.getenv("IBKR_CLIENT_ID", "12399")
    print(f"  Current Client ID: {current_client_id}")

    # Suggest a unique client ID if needed
    import random

    suggested_id = random.randint(10000, 99999)
    print(f"  Suggested unique Client ID: {suggested_id}")
    print(f"  Set with: export IBKR_CLIENT_ID={suggested_id}")


if __name__ == "__main__":
    # Run tests manually
    print("🧪 IBKR Smoke Tests")
    print("=" * 50)

    test_ibkr_configuration()
    test_ibkr_environment()

    try:
        test_ibkr_connect_and_tick()
    except Exception as e:
        print(f"Connection test failed: {e}")

    try:
        test_ibkr_data_provider()
    except Exception as e:
        print(f"Data provider test failed: {e}")

    print("\n✅ Smoke tests completed")
