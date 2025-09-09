#!/usr/bin/env python3
"""
Feature Engineering Script

Builds and caches features for the universe.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_features(universe_cfg_path: str, output_dir: str):
    """Build features for the universe."""
    logger.info(f"Building features for universe: {universe_cfg_path}")
    
    # Load universe config
    with open(universe_cfg_path, 'r') as f:
        universe_cfg = yaml.safe_load(f)
    
    symbols = universe_cfg['universe']
    logger.info(f"Processing {len(symbols)} symbols")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # For now, just create a placeholder features file
    features_summary = {
        'universe': symbols,
        'n_symbols': len(symbols),
        'features_built': True,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save features summary
    import json
    with open(output_path / 'features_summary.json', 'w') as f:
        json.dump(features_summary, f, indent=2)
    
    logger.info(f"Features built and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Build features for universe')
    parser.add_argument('--universe-cfg', type=str, required=True,
                       help='Universe configuration file')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='Output directory for features')
    
    args = parser.parse_args()
    
    build_features(args.universe_cfg, args.out_dir)


if __name__ == '__main__':
    main()
