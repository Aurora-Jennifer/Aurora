#!/usr/bin/env python3
"""
Secure Build Script
Compiles critical trading modules with Nuitka to protect source code.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        sys.exit(1)


def build_core_modules():
    """Build core modules with Nuitka."""

    # Create build directory
    build_dir = Path("core_compiled")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # Core modules to compile
    modules = [
        "core/engine/composer_integration.py",
        "core/composer/registry.py",
        "core/data_sanity.py",
        "core/config.py",
        "core/config_loader.py",
    ]

    for module in modules:
        if not Path(module).exists():
            print(f"⚠️  Module not found: {module}")
            continue

        module_name = Path(module).stem

        cmd = [
            "python",
            "-m",
            "nuitka",
            "--module",
            module,
            "--output-dir",
            str(build_dir),
            "--output-filename",
            f"{module_name}.so",
            "--nofollow-imports",
            "--include-data-files=config/*.yaml=config/",
            "--include-data-files=config/*.json=config/",
            "--remove-output",
            "--assume-yes-for-downloads",
        ]

        run_command(cmd, f"Compiling {module_name}")

    # Create __init__.py for the compiled package
    init_content = '''"""
Compiled Core Modules
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

# Import compiled modules
try:
    from . import composer_integration
    from . import registry
    from . import data_sanity
    from . import config
    from . import config_loader
except ImportError as e:
    print(f"Warning: Could not import compiled modules: {e}")
    print("Falling back to source modules...")
'''

    with open(build_dir / "__init__.py", "w") as f:
        f.write(init_content)


def build_api_client():
    """Build a simple API client for external use."""

    client_code = '''"""
Trading System API Client
Simple client for the trading system API.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import requests
from typing import Dict, List, Optional
import json

class TradingAPIClient:
    """Client for the trading system API."""
    
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def predict_signal(self, market_data: Dict) -> Dict:
        """Get trading signal prediction."""
        url = f"{self.base_url}/predict"
        response = requests.post(url, json=market_data, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def validate_data(self, market_data: Dict) -> Dict:
        """Validate market data."""
        url = f"{self.base_url}/validate"
        response = requests.post(url, json=market_data, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_capabilities(self) -> Dict:
        """Get system capabilities."""
        url = f"{self.base_url}/capabilities"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health."""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    client = TradingAPIClient("http://localhost:8000", "demo_token_placeholder")
    
    # Example market data
    market_data = {
        "market_data": {
            "symbol": "SPY",
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000000, 1000000, 1000000],
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"]
        }
    }
    
    try:
        # Test prediction
        result = client.predict_signal(market_data)
        print(f"Prediction: {result}")
        
        # Test validation
        validation = client.validate_data(market_data)
        print(f"Validation: {validation}")
        
    except Exception as e:
        print(f"Error: {e}")
'''

    with open("trading_api_client.py", "w") as f:
        f.write(client_code)


def create_demo_script():
    """Create a demo script for showcasing capabilities."""

    demo_script = '''#!/usr/bin/env python3
"""
Trading System Demo
Demonstrates the trading system capabilities via API.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import requests
import json
from datetime import datetime, timedelta
import random

def generate_sample_data(symbol: str, days: int = 30):
    """Generate sample market data for demonstration."""
    base_price = 100.0
    data = {
        "market_data": {
            "symbol": symbol,
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "timestamps": []
        }
    }
    
    current_price = base_price
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        # Generate realistic price movements
        change = random.uniform(-0.02, 0.02)  # ±2% daily change
        current_price *= (1 + change)
        
        # OHLC data
        open_price = current_price
        high_price = current_price * random.uniform(1.0, 1.01)
        low_price = current_price * random.uniform(0.99, 1.0)
        close_price = current_price * random.uniform(0.995, 1.005)
        
        data["market_data"]["open"].append(round(open_price, 2))
        data["market_data"]["high"].append(round(high_price, 2))
        data["market_data"]["low"].append(round(low_price, 2))
        data["market_data"]["close"].append(round(close_price, 2))
        data["market_data"]["volume"].append(random.randint(1000000, 5000000))
        data["market_data"]["timestamps"].append(
            (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        )
    
    return data

def run_demo():
    """Run the trading system demo."""
    print("🚀 Advanced Trading System Demo")
    print("=" * 50)
    
    # API configuration
    base_url = "http://localhost:8000"
    api_token = "demo_token_placeholder"
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Health check
        print("1. Health Check...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print("   ❌ API health check failed")
            return
        
        # Get capabilities
        print("\\n2. System Capabilities...")
        response = requests.get(f"{base_url}/capabilities", headers=headers)
        if response.status_code == 200:
            capabilities = response.json()
            print(f"   Features: {', '.join(capabilities['features'])}")
            print(f"   Supported assets: {', '.join(capabilities['supported_assets'])}")
        else:
            print("   ❌ Could not retrieve capabilities")
            return
        
        # Generate sample data
        print("\\n3. Generating Sample Data...")
        sample_data = generate_sample_data("SPY", days=30)
        print(f"   Generated {len(sample_data['market_data']['close'])} days of data")
        
        # Test prediction
        print("\\n4. Signal Prediction...")
        response = requests.post(f"{base_url}/predict", 
                               json=sample_data, headers=headers)
        if response.status_code == 200:
            prediction = response.json()
            print(f"   Signal: {prediction['signal']:.4f}")
            print(f"   Confidence: {prediction['confidence']:.2f}")
            print(f"   Regime: {prediction['regime']}")
            print(f"   Processing time: {prediction['processing_time_ms']:.2f}ms")
        else:
            print("   ❌ Prediction failed")
        
        # Test validation
        print("\\n5. Data Validation...")
        response = requests.post(f"{base_url}/validate", 
                               json=sample_data, headers=headers)
        if response.status_code == 200:
            validation = response.json()
            print(f"   Valid: {validation['is_valid']}")
            if validation['issues']:
                print(f"   Issues: {', '.join(validation['issues'])}")
            if validation['repairs']:
                print(f"   Repairs: {', '.join(validation['repairs'])}")
            print(f"   Validation time: {validation['validation_time_ms']:.2f}ms")
        else:
            print("   ❌ Validation failed")
        
        print("\\n✅ Demo completed successfully!")
        print("\\nThis demonstrates the trading system's capabilities")
        print("without exposing the underlying algorithms or source code.")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    run_demo()
'''

    with open("demo_trading_system.py", "w") as f:
        f.write(demo_script)

    # Make it executable
    os.chmod("demo_trading_system.py", 0o755)


def main():
    """Main build process."""
    print("🔒 Building Secure Trading System")
    print("=" * 50)

    # Check if Nuitka is installed
    try:
        import nuitka

        print("✅ Nuitka is available")
    except ImportError:
        print("❌ Nuitka not found. Installing...")
        run_command([sys.executable, "-m", "pip", "install", "nuitka"], "Installing Nuitka")

    # Build core modules
    build_core_modules()

    # Build API client
    build_api_client()

    # Create demo script
    create_demo_script()

    print("\\n✅ Secure build completed!")
    print("\\nGenerated files:")
    print("  - core_compiled/ (compiled modules)")
    print("  - trading_api_client.py (API client)")
    print("  - demo_trading_system.py (demo script)")
    print("  - api/demo_server.py (API server)")
    print("\\nTo run the demo:")
    print("  1. Start the API server: python api/demo_server.py")
    print("  2. Run the demo: python demo_trading_system.py")
    print("\\nSecurity features:")
    print("  - Core algorithms compiled to binary")
    print("  - API token authentication")
    print("  - No source code exposure")
    print("  - Canary strings for tracking")


if __name__ == "__main__":
    main()
