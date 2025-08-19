# tests/brokers/test_ibkr_unit.py
from datetime import datetime

import pytest

from brokers.ibkr import IBKRBroker, IBKRConfig


class _Order:  # minimal ib_insync-like placeholder
    def __init__(self, orderId, orderRef=None):
        self.orderId = orderId
        self.orderRef = orderRef


class _Trade:
    def __init__(self, order):
        self.order = order


class _Contract:
    def __init__(self, symbol):
        self.symbol = symbol
        self.localSymbol = symbol


class _Summary:
    def __init__(self, tag, value, currency):
        self.tag, self.value, self.currency = tag, value, currency


class FakeIB:
    def __init__(self):
        self._connected = False
        self._order_seq = 1000
        self._trades = []
        self._positions = []
        self._summary = []

    def connect(self, *a, **k):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def sleep(self, *a, **k):
        pass

    def placeOrder(self, contract, order):
        self._order_seq += 1
        t = _Trade(_Order(self._order_seq, orderRef=str(self._order_seq)))
        self._trades.append(t)
        return t

    def trades(self):
        return list(self._trades)

    def cancelOrder(self, order):  # order has orderId
        # mark canceled by simply keeping presence (adapter returns CANCEL_SENT)
        return None

    def positions(self, account=None):
        return list(self._positions)

    def accountSummary(self, account=None):
        return list(self._summary)


@pytest.fixture
def fake_ib(monkeypatch):
    from brokers import ibkr as mod

    fake = FakeIB()

    # monkeypatch ib_insync symbols inside module
    def _mk_ib():
        return fake

    mod.IB = _mk_ib

    def _mk_stock(symbol, exchange=None, currency=None):
        return _Contract(symbol)

    def _mk_mkt(**kw):
        return object()

    def _mk_lmt(**kw):
        return object()

    mod.Stock = _mk_stock
    mod.MarketOrder = _mk_mkt
    mod.LimitOrder = _mk_lmt

    return fake


def test_submit_market_and_cancel(fake_ib):
    broker = IBKRBroker(IBKRConfig())
    # Submit
    res = broker.submit_order("SPY", "BUY", 10, "market")
    assert res["status"] == "ACK"
    assert res["broker_order_id"] >= 1001
    # Cancel
    cres = broker.cancel_order(res["broker_order_id"])
    assert cres["status"] in ("CANCEL_SENT", "NOT_FOUND")


def test_positions_and_cash(fake_ib):
    # Seed positions and summary
    class _Pos:
        def __init__(self, symbol, position):
            self.contract = _Contract(symbol)
            self.position = position

    fake_ib._positions = [_Pos("SPY", 5), _Pos("TSLA", -2)]
    fake_ib._summary = [
        _Summary("TotalCashValue", "100000", "USD"),
        _Summary("NetLiquidation", "123456", "USD"),
    ]

    broker = IBKRBroker(IBKRConfig())
    pos = broker.get_positions()
    assert pos == {"SPY": 5.0, "TSLA": -2.0}

    cash = broker.get_cash()
    assert cash == 100000.0


def test_limit_order_qty_rounding(fake_ib):
    cfg = IBKRConfig(allow_fractional=False, qty_min=1.0)
    broker = IBKRBroker(cfg)
    r = broker.submit_order("AAPL", "SELL", 0.7, "limit")
    assert r["status"] == "ACK"


def test_fractional_shares_enabled(fake_ib):
    cfg = IBKRConfig(allow_fractional=True, qty_min=0.1)
    broker = IBKRBroker(cfg)
    r = broker.submit_order("AAPL", "BUY", 0.5, "market")
    assert r["status"] == "ACK"


def test_get_fills(fake_ib):
    # Mock a filled trade
    class _OrderStatus:
        def __init__(self):
            self.status = "Filled"
            self.filled = 10
            self.avgFillPrice = 150.0
            self.updateTime = "20240101 10:30:00"

    class _FilledTrade:
        def __init__(self):
            self.contract = _Contract("SPY")
            self.order = _Order(1001, "ref_1001")
            self.orderStatus = _OrderStatus()

    fake_ib._trades = [_FilledTrade()]

    broker = IBKRBroker(IBKRConfig())
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0]["symbol"] == "SPY"
    assert fills[0]["qty"] == 10.0
    assert fills[0]["price"] == 150.0


def test_broker_protocol_compliance(fake_ib):
    """Test that IBKRBroker implements the Broker protocol correctly."""
    broker = IBKRBroker(IBKRConfig())

    # Test submit_order method signature
    result = broker.submit_order("SPY", "BUY", 10, "market")
    assert "order_id" in result
    assert "status" in result
    assert "timestamp" in result

    # Test cancel_order method signature
    cancel_result = broker.cancel_order("1001")
    assert "status" in cancel_result
    assert "timestamp" in cancel_result

    # Test get_positions method signature
    positions = broker.get_positions()
    assert isinstance(positions, dict)

    # Test get_cash method signature
    cash = broker.get_cash()
    assert isinstance(cash, float)

    # Test now method signature
    now = broker.now()
    assert isinstance(now, datetime)
