#!/usr/bin/env python3
"""
Test Alpaca integration and update panel builder to use real data.
"""
import sys
import os
sys.path.append('.')

import pandas as pd
from datetime import datetime, timedelta
from ml.alpaca_data_provider import AlpacaDataProvider


def main():
    """Test Alpaca integration and validate for paper trading."""
    print("🚀 ALPACA PAPER TRADING INTEGRATION TEST")
    print("="*50)
    
    # Set environment variables
    os.environ['ALPACA_API_KEY'] = 'PKQ9ZKNTB5HV9SNQ929E'
    os.environ['ALPACA_SECRET_KEY'] = 'HaZ9FkKaXJdK1HFxp6Vr3449nMXUgPWvbyZhMpPn'
    os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets/v2'
    os.environ['IS_PAPER_TRADING'] = 'true'
    
    try:
        # Test Alpaca provider
        provider = AlpacaDataProvider()
        
        # Test 1: Account access
        print("\n📊 Step 1: Testing account access...")
        account = provider.get_account_info()
        
        if account:
            print(f"✅ Paper trading account connected")
            print(f"   Account Equity: ${account['equity']:,.2f}")
            print(f"   Buying Power: ${account['buying_power']:,.2f}")
            print(f"   Cash: ${account['cash']:,.2f}")
        else:
            print("❌ Failed to connect to account")
            return False
        
        # Test 2: Market data
        print("\n📈 Step 2: Testing market data retrieval...")
        
        # Test with a small universe
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = provider.get_daily_bars(test_symbols, start_date, end_date)
        
        if not data.empty:
            print(f"✅ Market data retrieved successfully")
            print(f"   Rows: {len(data)}")
            print(f"   Symbols: {sorted(data['symbol'].unique())}")
            print(f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}")
            
            # Show sample data
            print(f"\n📋 Sample data:")
            print(data.head().to_string(index=False))
            
        else:
            print("❌ No market data retrieved")
            return False
        
        # Test 3: Data quality validation
        print(f"\n🔍 Step 3: Validating data quality...")
        quality = provider.validate_data_quality(data)
        
        if quality['is_valid']:
            print("✅ Data quality validation passed")
        else:
            print("⚠️ Data quality issues found:")
            for issue in quality['issues']:
                print(f"   - {issue}")
        
        print(f"   Data age: {quality['checks']['data_age_hours']:.1f} hours")
        print(f"   Missing values: {sum(quality['checks']['missing_data'].values())}")
        
        # Test 4: Position tracking
        print(f"\n💼 Step 4: Testing position tracking...")
        positions = provider.get_positions()
        print(f"✅ Current positions: {len(positions)}")
        
        if len(positions) > 0:
            print(positions.to_string(index=False))
        else:
            print("   (No current positions - ready for fresh start)")
        
        # Test 5: Paper trading readiness
        print(f"\n🎯 Step 5: Paper trading readiness check...")
        
        readiness_checks = {
            'Account connected': account is not None,
            'Market data available': not data.empty,
            'Data quality valid': quality['is_valid'],
            'Recent data (< 3 days)': quality['checks']['data_age_hours'] < 72,
            'Multiple symbols': len(data['symbol'].unique()) >= 3,
            'Sufficient history': len(data) >= 50  # At least 50 bars total
        }
        
        all_ready = True
        for check, passed in readiness_checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
            if not passed:
                all_ready = False
        
        # Summary
        print(f"\n🏆 INTEGRATION TEST SUMMARY:")
        print("="*40)
        
        if all_ready:
            print("✅ READY FOR PAPER TRADING!")
            print("   Your Alpaca integration is working perfectly")
            print("   All systems green for 20-day validation")
            
            print(f"\n📋 Next steps:")
            print("   1. Update ml/panel_builder.py to use AlpacaDataProvider")
            print("   2. Run: ./daily_paper_trading.sh full")
            print("   3. Start automated paper trading validation")
            
        else:
            print("⚠️ SETUP ISSUES DETECTED")
            print("   Please resolve the failed checks above")
            print("   Then re-run this test")
        
        return all_ready
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Verify Alpaca API credentials are correct")
        print("2. Check network connectivity")
        print("3. Ensure paper trading account is active")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
