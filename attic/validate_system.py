#!/usr/bin/env python3
"""
System Validation Script
"""

import os


def validate_system():
    """Validate all system components."""
    print("🔍 System Validation Report")
    print("=" * 50)

    # Check IBKR connection
    try:
        from brokers.ibkr_broker import IBKRBroker, IBKRConfig

        config = IBKRConfig()
        config.port = 7497
        config.client_id = 12399
        broker = IBKRBroker(config=config, auto_connect=True)
        if broker.is_connected():
            print("✅ IBKR Connection: Working")
        else:
            print("❌ IBKR Connection: Failed")
        broker.disconnect()
    except Exception as e:
        print(f"❌ IBKR Connection: Error - {e}")

    # Check data provider
    try:
        from brokers.data_provider import IBKRDataProvider

        provider = IBKRDataProvider()
        data = provider.get_daily_data("SPY", "2025-08-01", "2025-08-13")
        if data is not None and len(data) > 0:
            print("✅ Data Provider: Working")
        else:
            print("❌ Data Provider: No data")
    except Exception as e:
        print(f"❌ Data Provider: Error - {e}")

    # Check strategy
    try:
        import yfinance as yf

        from strategies.regime_aware_ensemble import (
            RegimeAwareEnsembleParams,
            RegimeAwareEnsembleStrategy,
        )

        spy = yf.download("SPY", start="2025-07-01", end="2025-08-13")
        params = RegimeAwareEnsembleParams()
        strategy = RegimeAwareEnsembleStrategy(params)
        signals = strategy.generate_signals(spy)
        if len(signals) > 0:
            print("✅ Strategy: Working")
        else:
            print("❌ Strategy: No signals")
    except Exception as e:
        print(f"❌ Strategy: Error - {e}")

    # Check logs
    log_files = [
        "logs/trading_bot.log",
        "logs/trades/trades_2025-08.log",
        "logs/performance/performance_2025-08.log",
    ]

    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"✅ Log File: {log_file}")
        else:
            print(f"❌ Log File: {log_file} - Missing")

    # Check results
    result_files = [
        "results/performance_report.json",
        "results/trade_history.csv",
        "results/daily_returns.csv",
    ]

    for result_file in result_files:
        if os.path.exists(result_file):
            print(f"✅ Result File: {result_file}")
        else:
            print(f"❌ Result File: {result_file} - Missing")

    print("=" * 50)
    print("Validation Complete!")


if __name__ == "__main__":
    validate_system()
