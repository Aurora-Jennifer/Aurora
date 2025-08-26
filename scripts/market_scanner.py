#!/usr/bin/env python3
"""
Simple Market Scanner - Find Best Trading Opportunities
=====================================================

Watches multiple symbols and identifies the best trading opportunities
based on current market conditions.

Usage:
    python scripts/market_scanner.py --symbols SPY QQQ TSLA AAPL --scan-interval 60
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.runtime import build_features


def download_market_data(symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
    """Download market data for multiple symbols."""
    data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", auto_adjust=False)
            
            if len(df) < 20:
                print(f"⚠️  {symbol}: Insufficient data ({len(df)} days)")
                continue
                
            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
                
            data[symbol] = df
            
        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}")
    
    return data


def calculate_market_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key market metrics for a symbol."""
    close = df['Close']
    volume = df['Volume']
    
    # Price metrics
    current_price = close.iloc[-1]
    price_change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
    price_change_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    
    # Volatility
    returns = close.pct_change().dropna()
    volatility_20d = returns.rolling(20).std().iloc[-1] * 100
    
    # Volume metrics
    avg_volume_20d = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
    
    # Technical indicators
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    price_vs_sma20 = (current_price / sma_20 - 1) * 100
    price_vs_sma50 = (current_price / sma_50 - 1) * 100
    
    # RSI (simplified)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
    
    return {
        "current_price": current_price,
        "price_change_1d": price_change_1d,
        "price_change_5d": price_change_5d,
        "volatility_20d": volatility_20d,
        "volume_ratio": volume_ratio,
        "price_vs_sma20": price_vs_sma20,
        "price_vs_sma50": price_vs_sma50,
        "rsi": rsi
    }


def calculate_opportunity_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate opportunity score based on market conditions."""
    score = 0.0
    reasons = []
    
    # Momentum scoring (positive)
    if metrics["price_change_1d"] > 0:
        score += 1.0
        reasons.append("positive_1d")
    
    if metrics["price_change_5d"] > 0:
        score += 1.5
        reasons.append("positive_5d")
    
    # Volume scoring (positive)
    if metrics["volume_ratio"] > 1.2:
        score += 1.0
        reasons.append("high_volume")
    
    # Technical scoring
    if metrics["price_vs_sma20"] > 0:
        score += 1.0
        reasons.append("above_sma20")
    
    if metrics["price_vs_sma50"] > 0:
        score += 1.5
        reasons.append("above_sma50")
    
    # RSI scoring (avoid extremes)
    if 30 < metrics["rsi"] < 70:
        score += 1.0
        reasons.append("healthy_rsi")
    elif metrics["rsi"] < 30:
        score += 2.0  # Oversold - potential bounce
        reasons.append("oversold")
    
    # Volatility scoring (moderate is good)
    if 1.0 < metrics["volatility_20d"] < 3.0:
        score += 1.0
        reasons.append("moderate_vol")
    
    return {
        "score": score,
        "reasons": reasons,
        "recommendation": "BUY" if score >= 4.0 else "HOLD" if score >= 2.0 else "AVOID"
    }


def scan_market(symbols: List[str], scan_interval: int = 60):
    """Continuously scan market for opportunities."""
    print(f"🔍 Market Scanner Started")
    print(f"📈 Watching: {', '.join(symbols)}")
    print(f"⏰ Scan interval: {scan_interval} seconds")
    print("=" * 80)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n🕐 {current_time} - Scanning market...")
            
            # Download data
            data = download_market_data(symbols)
            
            if not data:
                print("❌ No data available")
                time.sleep(scan_interval)
                continue
            
            # Analyze each symbol
            opportunities = []
            
            for symbol, df in data.items():
                try:
                    # Calculate metrics
                    metrics = calculate_market_metrics(df)
                    
                    # Calculate opportunity score
                    opportunity = calculate_opportunity_score(metrics)
                    
                    # Store results
                    result = {
                        "symbol": symbol,
                        "metrics": metrics,
                        "opportunity": opportunity
                    }
                    opportunities.append(result)
                    
                    # Print summary
                    score = opportunity["score"]
                    rec = opportunity["recommendation"]
                    price = metrics["current_price"]
                    change_1d = metrics["price_change_1d"]
                    
                    status_icon = "🟢" if rec == "BUY" else "🟡" if rec == "HOLD" else "🔴"
                    print(f"{status_icon} {symbol:6} | ${price:6.2f} | {change_1d:+5.1f}% | Score: {score:4.1f} | {rec}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {symbol}: {e}")
            
            # Find best opportunities
            if opportunities:
                # Sort by score
                opportunities.sort(key=lambda x: x["opportunity"]["score"], reverse=True)
                
                print("\n🏆 TOP OPPORTUNITIES:")
                print("-" * 50)
                
                for i, opp in enumerate(opportunities[:3]):
                    symbol = opp["symbol"]
                    score = opp["opportunity"]["score"]
                    rec = opp["opportunity"]["recommendation"]
                    reasons = ", ".join(opp["opportunity"]["reasons"])
                    
                    print(f"{i+1}. {symbol} ({rec}) - Score: {score:.1f}")
                    print(f"   Reasons: {reasons}")
                
                # Save results
                results_file = Path("reports") / f"market_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                results_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(results_file, 'w') as f:
                    json.dump({
                        "timestamp": current_time,
                        "opportunities": opportunities
                    }, f, indent=2, default=str)
            
            print(f"\n⏳ Waiting {scan_interval} seconds until next scan...")
            time.sleep(scan_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")
            break
        except Exception as e:
            print(f"❌ Scanner error: {e}")
            time.sleep(scan_interval)


def quick_scan(symbols: List[str]):
    """Do a single market scan."""
    print(f"🔍 Quick Market Scan - {', '.join(symbols)}")
    print("=" * 60)
    
    data = download_market_data(symbols)
    
    if not data:
        print("❌ No data available")
        return
    
    opportunities = []
    
    for symbol, df in data.items():
        try:
            metrics = calculate_market_metrics(df)
            opportunity = calculate_opportunity_score(metrics)
            
            result = {
                "symbol": symbol,
                "metrics": metrics,
                "opportunity": opportunity
            }
            opportunities.append(result)
            
            # Print detailed analysis
            print(f"\n📊 {symbol} Analysis:")
            print(f"   Price: ${metrics['current_price']:.2f}")
            print(f"   1D Change: {metrics['price_change_1d']:+.2f}%")
            print(f"   5D Change: {metrics['price_change_5d']:+.2f}%")
            print(f"   Volatility: {metrics['volatility_20d']:.2f}%")
            print(f"   Volume Ratio: {metrics['volume_ratio']:.2f}x")
            print(f"   vs SMA20: {metrics['price_vs_sma20']:+.2f}%")
            print(f"   vs SMA50: {metrics['price_vs_sma50']:+.2f}%")
            print(f"   RSI: {metrics['rsi']:.1f}")
            print(f"   Score: {opportunity['score']:.1f} ({opportunity['recommendation']})")
            print(f"   Reasons: {', '.join(opportunity['reasons'])}")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    # Show summary
    if opportunities:
        print(f"\n🏆 SUMMARY:")
        print("-" * 40)
        
        opportunities.sort(key=lambda x: x["opportunity"]["score"], reverse=True)
        
        for i, opp in enumerate(opportunities):
            symbol = opp["symbol"]
            score = opp["opportunity"]["score"]
            rec = opp["opportunity"]["recommendation"]
            
            status_icon = "🟢" if rec == "BUY" else "🟡" if rec == "HOLD" else "🔴"
            print(f"{i+1}. {status_icon} {symbol} - {rec} (Score: {score:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Market Scanner")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "TSLA", "AAPL"],
                       help="Symbols to scan")
    parser.add_argument("--scan-interval", type=int, default=60,
                       help="Scan interval in seconds")
    parser.add_argument("--quick", action="store_true",
                       help="Do a single quick scan instead of continuous")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_scan(args.symbols)
    else:
        scan_market(args.symbols, args.scan_interval)


if __name__ == "__main__":
    main()
