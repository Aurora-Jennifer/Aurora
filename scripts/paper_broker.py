import argparse
import json
import os
import time
from pathlib import Path


class HoldingsLedger:
    """Track positions and P&L across symbols"""

    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: dict[str, float] = {}  # symbol -> quantity
        self.avg_prices: dict[str, float] = {}  # symbol -> average price
        self.trades: list[dict] = []

    def execute_trade(self, symbol: str, quantity: float, price: float,
                     commission_bps: float = 0.3, slippage_bps: float = 0.7) -> dict:
        """Execute a trade and update holdings"""
        # Calculate costs
        trade_value = abs(quantity * price)
        commission = trade_value * (commission_bps / 10000.0)
        slippage = trade_value * (slippage_bps / 10000.0)
        total_cost = trade_value + commission + slippage

        # Update cash
        if quantity > 0:  # Buy
            if self.cash < total_cost:
                return {"status": "rejected", "reason": "insufficient_cash"}
            self.cash -= total_cost
        else:  # Sell
            self.cash += trade_value - commission - slippage

        # Update positions
        current_qty = self.positions.get(symbol, 0.0)
        new_qty = current_qty + quantity

        if new_qty == 0:
            # Close position
            if symbol in self.positions:
                del self.positions[symbol]
            if symbol in self.avg_prices:
                del self.avg_prices[symbol]
        else:
            # Update position
            self.positions[symbol] = new_qty

            # Update average price (VWAP)
            if current_qty == 0:
                self.avg_prices[symbol] = price
            else:
                # VWAP calculation
                current_value = current_qty * self.avg_prices[symbol]
                new_value = quantity * price
                total_value = current_value + new_value
                total_qty = current_qty + quantity
                self.avg_prices[symbol] = total_value / total_qty

        # Record trade
        trade = {
            "timestamp": time.time(),
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "slippage": slippage,
            "total_cost": total_cost,
            "cash_after": self.cash,
            "position_after": new_qty
        }
        self.trades.append(trade)

        return {"status": "filled", "trade": trade}

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value"""
        portfolio_value = self.cash
        for symbol, qty in self.positions.items():
            if symbol in current_prices:
                portfolio_value += qty * current_prices[symbol]
        return portfolio_value

    def get_position_summary(self) -> dict:
        """Get current position summary"""
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "avg_prices": dict(self.avg_prices),
            "total_trades": len(self.trades)
        }


def should_kill(kill_file: Path) -> bool:
    if os.getenv("SERVE_DUMMY", "0") == "1":
        return False
    return bool(kill_file.exists())


def calculate_position_size(score: float, rank: float | None,
                          max_position_pct: float = 0.15,
                          base_position_size: float = 1000.0) -> float:
    """Calculate position size based on signal strength and limits"""
    # Use rank if available, otherwise use score
    signal_strength = rank if rank is not None else abs(score)

          # Normalize to 0-1 range
    normalized = signal_strength if rank is not None else (signal_strength + 1) / 2

    # Apply position sizing
    position_size = base_position_size * normalized

    # Apply position limits
    max_position = base_position_size * max_position_pct
    return min(position_size, max_position)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="Input signals JSONL (ranked or scores)")
    ap.add_argument("--fills", required=True, help="Output fills JSONL path")
    ap.add_argument("--kill", default="kill.flag", help="Kill-switch file path")
    ap.add_argument("--sleep-ms", type=int, default=50)
    ap.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash")
    ap.add_argument("--commission-bps", type=float, default=0.3, help="Commission in basis points")
    ap.add_argument("--slippage-bps", type=float, default=0.7, help="Slippage in basis points")
    ap.add_argument("--max-position-pct", type=float, default=0.15, help="Max position as percent of portfolio")
    ap.add_argument("--base-position-size", type=float, default=1000.0, help="Base position size")
    ap.add_argument("--ledger", help="Output ledger JSON path")
    args = ap.parse_args()

    sig_path = Path(args.signals)
    fills_path = Path(args.fills)
    kill_path = Path(args.kill)
    fills_path.parent.mkdir(parents=True, exist_ok=True)

    if not sig_path.exists():
        print(f"[PAPER][FAIL] signals not found: {sig_path}")
        return 1

    # Initialize holdings ledger
    ledger = HoldingsLedger(initial_cash=args.initial_cash)

    # Track current prices for portfolio valuation
    current_prices: dict[str, float] = {}

    with open(sig_path) as fin, open(fills_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            if should_kill(kill_path):
                print("[PAPER][HALT] kill-switch engaged; stopping")
                break
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                print("[PAPER][WARN] skipping malformed line")
                continue

            symbol = s.get("symbol", "Unknown")
            score = s.get("score", 0.0)
            rank = s.get("rank")
            price = s.get("price", 100.0)  # Default price if not provided

            # Update current price
            current_prices[symbol] = price

            # Calculate position size
            position_size = calculate_position_size(
                score=score,
                rank=rank,
                max_position_pct=args.max_position_pct,
                base_position_size=args.base_position_size
            )

            # Determine trade direction (simplified: positive score = buy, negative = sell)
            quantity = position_size / price if score > 0 else -position_size / price

            # Execute trade
            result = ledger.execute_trade(
                symbol=symbol,
                quantity=quantity,
                price=price,
                commission_bps=args.commission_bps,
                slippage_bps=args.slippage_bps
            )

            # Create fill record
            fill = {
                "ts": time.time(),
                "symbol": symbol,
                "score": score,
                "rank": rank,
                "quantity": quantity,
                "price": price,
                "status": result["status"],
                "commission": result.get("trade", {}).get("commission", 0.0),
                "slippage": result.get("trade", {}).get("slippage", 0.0),
                "cash_after": ledger.cash,
                "portfolio_value": ledger.get_portfolio_value(current_prices)
            }

            fout.write(json.dumps(fill) + "\n")
            fout.flush()
            time.sleep(args.sleep_ms / 1000.0)

    # Save ledger summary
    if args.ledger:
        ledger_path = Path(args.ledger)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "final_summary": ledger.get_position_summary(),
            "portfolio_value": ledger.get_portfolio_value(current_prices),
            "total_return_pct": ((ledger.get_portfolio_value(current_prices) / args.initial_cash) - 1) * 100,
            "total_trades": len(ledger.trades),
            "trades": ledger.trades
        }

        with open(ledger_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"[PAPER][OK] fills written -> {fills_path}")
    print(f"[PAPER][OK] final portfolio value: ${ledger.get_portfolio_value(current_prices):,.2f}")
    print(f"[PAPER][OK] total return: {((ledger.get_portfolio_value(current_prices) / args.initial_cash) - 1) * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


