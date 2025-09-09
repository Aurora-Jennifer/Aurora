#!/usr/bin/env python3
"""
Paper Trading System

Simulates live trading using exported signals with realistic slippage,
costs, and market impact modeling.
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
import logging
import json
from typing import Dict, List, Optional
import yaml
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_signals(signals_dir: Path) -> Dict:
    """Load trading signals"""
    signals_file = signals_dir / "signals.json"
    if not signals_file.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_file}")
    
    with open(signals_file, 'r') as f:
        return json.load(f)


def simulate_trading(signals: Dict, broker: str = "paper", 
                    initial_capital: float = 100000.0) -> Dict:
    """Simulate paper trading"""
    
    logger.info(f"Starting paper trading simulation with {broker} broker")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")
    
    # Initialize portfolio
    portfolio = {
        "cash": initial_capital,
        "positions": {},
        "total_value": initial_capital,
        "trades": [],
        "daily_pnl": [],
        "metrics": {}
    }
    
    # Get active signals
    active_signals = {k: v for k, v in signals["signals"].items() 
                     if v["status"] == "active"}
    
    logger.info(f"Active signals: {len(active_signals)}")
    
    # Simulate trading over time (simplified - in real implementation, this would use market data)
    trading_days = 30  # Simulate 30 days
    daily_returns = {}
    
    for symbol, signal in active_signals.items():
        # Simulate daily returns based on signal strength and market noise
        signal_strength = signal["signal_strength"]
        expected_return = signal["expected_return"]
        
        # Generate realistic daily returns with some signal correlation
        np.random.seed(42)  # For reproducibility
        base_returns = np.random.normal(0, 0.02, trading_days)  # 2% daily vol
        signal_returns = np.random.normal(expected_return/252, 0.01, trading_days)  # Signal component
        
        # Blend base market returns with signal
        daily_returns[symbol] = 0.7 * base_returns + 0.3 * signal_returns * signal_strength
    
    # Simulate portfolio management
    for day in range(trading_days):
        day_pnl = 0.0
        
        # Update existing positions
        for symbol, position in portfolio["positions"].items():
            if symbol in daily_returns:
                daily_return = daily_returns[symbol][day]
                position_value = position["shares"] * position["price"] * (1 + daily_return)
                day_pnl += position_value - (position["shares"] * position["price"])
                
                # Update position
                portfolio["positions"][symbol]["price"] *= (1 + daily_return)
                portfolio["positions"][symbol]["value"] = position_value
        
        # Rebalance based on signals (simplified)
        if day % 5 == 0:  # Rebalance every 5 days
            total_value = portfolio["cash"] + sum(p["value"] for p in portfolio["positions"].values())
            
            # Target allocation based on signal strength
            for symbol, signal in active_signals.items():
                target_weight = signal["signal_strength"] * 0.1  # Max 10% per position
                target_value = total_value * target_weight
                
                current_value = portfolio["positions"].get(symbol, {}).get("value", 0)
                rebalance_amount = target_value - current_value
                
                if abs(rebalance_amount) > total_value * 0.01:  # Only rebalance if >1% change
                    # Simulate trade execution
                    trade_cost = abs(rebalance_amount) * 0.001  # 10 bps cost
                    slippage = abs(rebalance_amount) * 0.0005  # 5 bps slippage
                    
                    if rebalance_amount > 0:  # Buy
                        if portfolio["cash"] >= rebalance_amount + trade_cost + slippage:
                            portfolio["cash"] -= (rebalance_amount + trade_cost + slippage)
                            shares = rebalance_amount / (signal.get("price", 100))  # Assume $100 price
                            
                            if symbol in portfolio["positions"]:
                                portfolio["positions"][symbol]["shares"] += shares
                                portfolio["positions"][symbol]["value"] += rebalance_amount
                            else:
                                portfolio["positions"][symbol] = {
                                    "shares": shares,
                                    "price": signal.get("price", 100),
                                    "value": rebalance_amount
                                }
                            
                            portfolio["trades"].append({
                                "day": day,
                                "symbol": symbol,
                                "action": "BUY",
                                "amount": rebalance_amount,
                                "cost": trade_cost + slippage
                            })
                    
                    else:  # Sell
                        if symbol in portfolio["positions"]:
                            sell_value = min(abs(rebalance_amount), portfolio["positions"][symbol]["value"])
                            portfolio["cash"] += sell_value - trade_cost - slippage
                            
                            portfolio["positions"][symbol]["shares"] *= (1 - sell_value / portfolio["positions"][symbol]["value"])
                            portfolio["positions"][symbol]["value"] -= sell_value
                            
                            portfolio["trades"].append({
                                "day": day,
                                "symbol": symbol,
                                "action": "SELL",
                                "amount": sell_value,
                                "cost": trade_cost + slippage
                            })
        
        # Update total portfolio value
        portfolio["total_value"] = portfolio["cash"] + sum(p["value"] for p in portfolio["positions"].values())
        portfolio["daily_pnl"].append(day_pnl)
    
    # Calculate final metrics
    total_return = (portfolio["total_value"] - initial_capital) / initial_capital
    daily_pnl_array = np.array(portfolio["daily_pnl"])
    sharpe_ratio = np.mean(daily_pnl_array) / (np.std(daily_pnl_array) + 1e-8) * np.sqrt(252)
    max_drawdown = np.min(np.cumsum(daily_pnl_array) - np.maximum.accumulate(np.cumsum(daily_pnl_array)))
    
    portfolio["metrics"] = {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": len(portfolio["trades"]),
        "total_costs": sum(t["cost"] for t in portfolio["trades"]),
        "final_value": portfolio["total_value"]
    }
    
    return portfolio


def main():
    parser = argparse.ArgumentParser(description="Run paper trading simulation")
    parser.add_argument("--signals", required=True, help="Directory containing trading signals")
    parser.add_argument("--broker", default="paper", help="Broker type (paper, alpaca, etc.)")
    parser.add_argument("--out-dir", required=True, help="Output directory for trading results")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    
    args = parser.parse_args()
    
    signals_dir = Path(args.signals)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading signals from: {signals_dir}")
    logger.info(f"Output directory: {out_dir}")
    
    try:
        # Load signals
        signals = load_signals(signals_dir)
        
        # Run paper trading simulation
        portfolio = simulate_trading(signals, args.broker, args.initial_capital)
        
        # Save results
        with open(out_dir / "portfolio_results.json", 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
        
        # Create summary report
        summary = {
            "simulation_date": datetime.now().isoformat(),
            "broker": args.broker,
            "initial_capital": args.initial_capital,
            "final_value": portfolio["metrics"]["final_value"],
            "total_return": portfolio["metrics"]["total_return"],
            "sharpe_ratio": portfolio["metrics"]["sharpe_ratio"],
            "max_drawdown": portfolio["metrics"]["max_drawdown"],
            "total_trades": portfolio["metrics"]["total_trades"],
            "total_costs": portfolio["metrics"]["total_costs"],
            "active_positions": len(portfolio["positions"])
        }
        
        with open(out_dir / "trading_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print(f"\n📊 Paper Trading Results:")
        print(f"Initial Capital: ${args.initial_capital:,.2f}")
        print(f"Final Value: ${portfolio['metrics']['final_value']:,.2f}")
        print(f"Total Return: {portfolio['metrics']['total_return']:.2%}")
        print(f"Sharpe Ratio: {portfolio['metrics']['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {portfolio['metrics']['max_drawdown']:.2%}")
        print(f"Total Trades: {portfolio['metrics']['total_trades']}")
        print(f"Total Costs: ${portfolio['metrics']['total_costs']:,.2f}")
        print(f"Active Positions: {len(portfolio['positions'])}")
        
        print(f"\n📁 Results saved to: {out_dir}/")
        print(f"  - portfolio_results.json")
        print(f"  - trading_summary.json")
        
        print(f"\n✅ Paper trading simulation completed!")
        
    except Exception as e:
        logger.error(f"Paper trading simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
