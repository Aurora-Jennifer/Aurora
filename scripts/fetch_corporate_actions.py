#!/usr/bin/env python3
"""
Fetch corporate actions (splits, dividends) from yfinance.
Part of Clearframe Rung 4 - Corporate Actions handling.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

# Configure logging
from core.utils import setup_logging
logger = setup_logging("logs/fetch_corporate_actions.log", logging.INFO)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def fetch_corporate_actions(symbol: str, logger) -> dict:
    """Fetch splits and dividends for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get splits and dividends
        splits = ticker.splits
        dividends = ticker.dividends
        
        actions_data = {
            "symbol": symbol,
            "splits": splits.to_dict() if not splits.empty else {},
            "dividends": dividends.to_dict() if not dividends.empty else {},
            "splits_count": len(splits),
            "dividends_count": len(dividends),
            "first_split": splits.index.min().isoformat() if not splits.empty else None,
            "last_split": splits.index.max().isoformat() if not splits.empty else None,
            "first_dividend": dividends.index.min().isoformat() if not dividends.empty else None,
            "last_dividend": dividends.index.max().isoformat() if not dividends.empty else None,
            "total_dividend_amount": dividends.sum() if not dividends.empty else 0.0,
            "largest_split_ratio": splits.max() if not splits.empty else None
        }
        
        logger.info(f"✅ {symbol}: {len(splits)} splits, {len(dividends)} dividends")
        if not splits.empty:
            logger.info(f"   📊 Splits: {splits.index.min().strftime('%Y-%m-%d')} to {splits.index.max().strftime('%Y-%m-%d')}")
        if not dividends.empty:
            logger.info(f"   💰 Dividends: ${dividends.sum():.2f} total, latest ${dividends.iloc[-1]:.2f}")
            
        return actions_data
        
    except Exception as e:
        logger.error(f"Failed to fetch corporate actions for {symbol}: {e}")
        return None


def save_actions_data(actions_data: dict, output_dir: Path, logger):
    """Save corporate actions data."""
    if not actions_data:
        return None
        
    symbol = actions_data["symbol"]
    
    # Save detailed data as JSON
    actions_file = output_dir / f"{symbol}_actions.json"
    with open(actions_file, 'w') as f:
        # Convert datetime keys to strings for JSON serialization
        actions_json = actions_data.copy()
        
        # Convert splits dict (datetime keys -> strings)
        if actions_json["splits"]:
            actions_json["splits"] = {
                pd.Timestamp(k).isoformat(): v 
                for k, v in actions_json["splits"].items()
            }
            
        # Convert dividends dict (datetime keys -> strings) 
        if actions_json["dividends"]:
            actions_json["dividends"] = {
                pd.Timestamp(k).isoformat(): v 
                for k, v in actions_json["dividends"].items()
            }
            
        json.dump(actions_json, f, indent=2)
    
    logger.info(f"💾 Saved {symbol} actions: {actions_file}")
    return actions_file


def generate_summary(all_actions: list[dict], output_dir: Path, logger):
    """Generate summary of corporate actions across all symbols."""
    summary_data = []
    
    for actions in all_actions:
        if actions:
            summary_data.append({
                "symbol": actions["symbol"],
                "splits_count": actions["splits_count"],
                "dividends_count": actions["dividends_count"],
                "total_dividend_amount": actions["total_dividend_amount"],
                "largest_split_ratio": actions["largest_split_ratio"],
                "first_split": actions["first_split"],
                "last_split": actions["last_split"],
                "first_dividend": actions["first_dividend"], 
                "last_dividend": actions["last_dividend"],
                "has_major_split": 1 if (actions["largest_split_ratio"] is not None and actions["largest_split_ratio"] >= 2.0) else 0,
                "dividend_yield_proxy": float(actions["total_dividend_amount"] if actions["total_dividend_amount"] > 0 else 0)
            })
    
    # Save summary
    summary_file = output_dir / "corporate_actions_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    summary_csv = output_dir / "corporate_actions_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    
    logger.info(f"📋 Generated summary: {summary_file}")
    logger.info(f"📋 Generated CSV summary: {summary_csv}")
    
    # Print top split events
    splits_df = pd.DataFrame([d for d in summary_data if d["splits_count"] > 0])
    if not splits_df.empty:
        logger.info("\n📊 Top Split Events:")
        top_splits = splits_df.nlargest(5, "largest_split_ratio")[["symbol", "largest_split_ratio", "splits_count"]]
        for _, row in top_splits.iterrows():
            logger.info(f"   {row['symbol']:<6} {row['largest_split_ratio']:>6.1f}:1 ratio ({row['splits_count']} total splits)")
    
    # Print top dividend payers
    div_df = pd.DataFrame([d for d in summary_data if d["dividends_count"] > 0])
    if not div_df.empty:
        logger.info("\n💰 Top Dividend Payers:")
        top_divs = div_df.nlargest(5, "total_dividend_amount")[["symbol", "total_dividend_amount", "dividends_count"]]
        for _, row in top_divs.iterrows():
            logger.info(f"   {row['symbol']:<6} ${row['total_dividend_amount']:>8.2f} total ({row['dividends_count']} payments)")


def main():
    parser = argparse.ArgumentParser(description="Fetch corporate actions from yfinance.")
    parser.add_argument("--symbols", nargs='+', 
                       default=["SPY", "QQQ", "IWM", "TLT", "GLD", "AAPL", "MSFT", "GOOGL", "BRK-B", "JNJ", "XOM"],
                       help="List of ticker symbols to fetch actions for.")
    parser.add_argument("--output-dir", type=str, default="data/corporate_actions",
                       help="Output directory for corporate actions data.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting corporate actions acquisition")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Output: {output_path}")

    all_actions = []
    successful_count = 0

    for symbol in args.symbols:
        logger.info(f"Fetching corporate actions for {symbol}...")
        actions_data = fetch_corporate_actions(symbol, logger)
        
        if actions_data:
            save_actions_data(actions_data, output_path, logger)
            all_actions.append(actions_data)
            successful_count += 1
        else:
            logger.warning(f"Failed to process {symbol}")

    logger.info(f"🎉 Completed: {successful_count}/{len(args.symbols)} symbols successful")
    
    generate_summary(all_actions, output_path, logger)
    
    logger.info(f"✅ Corporate actions data ready at: {output_path}")
    logger.info(f"   Next: Implement split/dividend adjustment logic")
    logger.info(f"   Next: Add DataSanity validation for corporate action periods")


if __name__ == "__main__":
    main()
