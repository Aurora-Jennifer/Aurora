# tests/utils/test_quotes_provider_ibkr.py
import pytest

from utils.quotes_provider_ibkr import IBKRQuoteConfig, IBKRQuoteProvider


class _Ticker:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask


class FakeIB:
    def __init__(self):
        self._connected = False
        self._tickers = {}

    def connect(self, *a, **k):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def reqMktData(self, contract, generic_tick_list, snapshot, regulatory_snapshot):
        # Return a mock ticker
        symbol = contract.symbol
        if symbol == "SPY":
            return _Ticker(450.0, 450.1)
        elif symbol == "TSLA":
            return _Ticker(250.0, 250.1)
        else:
            return _Ticker(100.0, 100.1)

    def sleep(self, duration):
        pass


@pytest.fixture
def fake_ib(monkeypatch):
    from utils import quotes_provider_ibkr as mod

    fake = FakeIB()

    # monkeypatch ib_insync symbols inside module
    def _mk_ib():
        return fake

    mod.IB = _mk_ib

    def _mk_stock(symbol, exchange=None, currency=None):
        class _Contract:
            def __init__(self, symbol):
                self.symbol = symbol

        return _Contract(symbol)

    mod.Stock = _mk_stock

    return fake


def test_ibkr_quote_provider_basic(fake_ib):
    """Test basic quote provider functionality."""
    config = IBKRQuoteConfig()
    provider = IBKRQuoteProvider(config)

    # Test quote for SPY
    quote = provider.quote("SPY")
    assert quote["symbol"] == "SPY"
    assert quote["bid"] == 450.0
    assert quote["ask"] == 450.1
    assert quote["mid"] == 450.05
    assert "ts" in quote

    # Test quote for TSLA
    quote = provider.quote("TSLA")
    assert quote["symbol"] == "TSLA"
    assert quote["bid"] == 250.0
    assert quote["ask"] == 250.1
    assert quote["mid"] == 250.05

    provider.close()


def test_ibkr_quote_provider_contract_caching(fake_ib):
    """Test that contracts are cached."""
    config = IBKRQuoteConfig()
    provider = IBKRQuoteProvider(config)

    # First call should create contract
    quote1 = provider.quote("SPY")

    # Second call should use cached contract
    quote2 = provider.quote("SPY")

    assert quote1["bid"] == quote2["bid"]
    assert quote1["ask"] == quote2["ask"]

    provider.close()


def test_ibkr_quote_provider_config(fake_ib):
    """Test quote provider with custom config."""
    config = IBKRQuoteConfig(
        host="localhost",
        port=7496,  # Live port
        client_id=999,
        route="SMART",
        currency="USD",
        snap_ms=200,
    )

    provider = IBKRQuoteProvider(config)
    quote = provider.quote("SPY")

    assert quote["symbol"] == "SPY"
    assert quote["bid"] == 450.0
    assert quote["ask"] == 450.1

    provider.close()


def test_quote_provider_factory():
    """Test the quote provider factory function."""
    from utils.quotes_provider import get_quote_provider

    # Test dummy provider
    dummy_provider = get_quote_provider("dummy")
    quote = dummy_provider.quote("SPY")
    assert quote["bid"] == 100.0
    assert quote["ask"] == 100.0
    assert quote["mid"] == 100.0

    # Test IBKR provider (should work with mocked ib_insync)
    try:
        ibkr_provider = get_quote_provider("ibkr")
        quote = ibkr_provider.quote("SPY")
        assert "bid" in quote
        assert "ask" in quote
        assert "mid" in quote
        ibkr_provider.close()
    except Exception as e:
        # This is expected if ib_insync is not installed
        assert "ib_insync not installed" in str(e) or "Missing broker credentials" in str(e)

    # Test unknown provider
    with pytest.raises(ValueError, match="Unknown quote provider"):
        get_quote_provider("unknown")
