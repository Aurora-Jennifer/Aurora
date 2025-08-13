#!/usr/bin/env python3
"""
Test IBKR Connection with Correct Settings
"""

from brokers.ibkr_broker import IBKRBroker, IBKRConfig


def test_connection():
    """Test IBKR connection with correct settings."""
    print("Testing IBKR Connection...")
    print("Settings: Port 7497, Client ID 12399, Host 127.0.0.1")

    # Create config with correct settings
    config = IBKRConfig()
    config.port = 7497
    config.client_id = 12399
    config.host = "127.0.0.1"

    try:
        # Create broker and connect
        broker = IBKRBroker(config=config, auto_connect=True)

        if broker.is_connected():
            print("✅ IBKR connection successful!")

            # Test account info
            try:
                account_info = broker.get_account_info()
                print(f"Account Info: {account_info}")
            except Exception as e:
                print(f"Warning: Could not get account info: {e}")

            # Test positions
            try:
                positions = broker.get_positions()
                print(f"Positions: {positions}")
            except Exception as e:
                print(f"Warning: Could not get positions: {e}")

            # Test market data
            try:
                spy_data = broker.get_market_data("SPY", "1 D", "1 hour")
                if spy_data is not None:
                    print(f"SPY Data: {len(spy_data)} rows")
                    print(f"Last Close: ${spy_data['Close'].iloc[-1]:.2f}")
                else:
                    print("No SPY data available")
            except Exception as e:
                print(f"Warning: Could not get market data: {e}")

            broker.disconnect()
            return True

        else:
            print("❌ Failed to connect to IBKR")
            return False

    except Exception as e:
        print(f"❌ Error connecting to IBKR: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 IBKR connection test passed!")
    else:
        print(
            "\n⚠️  IBKR connection test failed. Check your IBKR Gateway configuration."
        )
