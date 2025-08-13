"""
Enhanced Paper Trading System - Main Entry Point
This file now serves as the main entry point, delegating to modular components.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from cli.paper import main

if __name__ == "__main__":
    main()
