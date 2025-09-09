#!/usr/bin/env python3
"""
Quick Alpaca API connectivity test to diagnose authentication issues.
"""
import alpaca_trade_api as tradeapi

def test_alpaca_auth():
    """Test basic Alpaca API authentication."""
    print("🔑 ALPACA API AUTHENTICATION TEST")
    print("="*40)
    
    # Your credentials
    api_key = "PK4LGVF9BG8IX0DHS7VO"
    secret_key = "Qt1fh7cS6UgfwDZpQx87YHiDXt3mOIYSdmu"
    
    # Test different endpoints
    endpoints = [
        ("https://paper-api.alpaca.markets", "Paper Trading API"),
        ("https://paper-api.alpaca.markets/v2", "Paper Trading API v2"),
        ("https://data.alpaca.markets", "Market Data API")
    ]
    
    for base_url, description in endpoints:
        print(f"\n📡 Testing: {description}")
        print(f"   URL: {base_url}")
        
        try:
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url,
                api_version="v2"
            )
            
            # Test account access
            account = api.get_account()
            
            print(f"✅ SUCCESS!")
            print(f"   Account ID: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Equity: ${float(account.equity):,.2f}")
            print(f"   Paper Trading: {account.pattern_day_trader}")
            
            return True
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            
            # Analyze the error
            error_str = str(e).lower()
            
            if "unauthorized" in error_str:
                print("   🔍 Diagnosis: API keys are invalid or expired")
                print("   💡 Solution: Check Alpaca dashboard, regenerate keys")
            elif "forbidden" in error_str:
                print("   🔍 Diagnosis: API keys valid but missing permissions")
                print("   💡 Solution: Enable paper trading in Alpaca account")
            elif "not found" in error_str:
                print("   🔍 Diagnosis: Wrong API endpoint")
                print("   💡 Solution: Check base URL")
            else:
                print("   🔍 Diagnosis: Unknown error")
                print("   💡 Solution: Check network connectivity")
    
    print(f"\n❌ ALL ENDPOINTS FAILED")
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Log into https://alpaca.markets/")
    print(f"2. Go to 'API Keys' section")
    print(f"3. Confirm paper trading is enabled")
    print(f"4. Generate new API keys if needed")
    print(f"5. Ensure keys have 'Trading' and 'Data' permissions")
    
    return False

if __name__ == "__main__":
    test_alpaca_auth()
