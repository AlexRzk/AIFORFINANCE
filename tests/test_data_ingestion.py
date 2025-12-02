"""
Tests for the data ingestion module.
"""
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.base import OrderBookSnapshot, Trade, PriceLevel


class TestPriceLevel:
    """Test PriceLevel dataclass."""
    
    def test_creation(self):
        level = PriceLevel(price=100.0, quantity=10.0)
        assert level.price == 100.0
        assert level.quantity == 10.0


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot dataclass."""
    
    def test_creation(self):
        bids = [PriceLevel(100.0, 10.0), PriceLevel(99.9, 20.0)]
        asks = [PriceLevel(100.1, 5.0), PriceLevel(100.2, 15.0)]
        
        snapshot = OrderBookSnapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            bids=bids,
            asks=asks,
        )
        
        assert snapshot.exchange == "binance"
        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
    
    def test_best_bid_ask(self):
        bids = [PriceLevel(100.0, 10.0), PriceLevel(99.9, 20.0)]
        asks = [PriceLevel(100.1, 5.0), PriceLevel(100.2, 15.0)]
        
        snapshot = OrderBookSnapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            bids=bids,
            asks=asks,
        )
        
        assert snapshot.best_bid.price == 100.0
        assert snapshot.best_ask.price == 100.1
    
    def test_mid_price(self):
        bids = [PriceLevel(100.0, 10.0)]
        asks = [PriceLevel(100.2, 5.0)]
        
        snapshot = OrderBookSnapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            bids=bids,
            asks=asks,
        )
        
        assert snapshot.mid_price == 100.1
    
    def test_spread(self):
        bids = [PriceLevel(100.0, 10.0)]
        asks = [PriceLevel(100.2, 5.0)]
        
        snapshot = OrderBookSnapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            bids=bids,
            asks=asks,
        )
        
        assert abs(snapshot.spread - 0.2) < 1e-10
    
    def test_to_dict(self):
        bids = [PriceLevel(100.0, 10.0)]
        asks = [PriceLevel(100.1, 5.0)]
        
        snapshot = OrderBookSnapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            bids=bids,
            asks=asks,
            sequence=12345,
        )
        
        d = snapshot.to_dict()
        
        assert d["exchange"] == "binance"
        assert d["symbol"] == "BTCUSDT"
        assert d["bids"] == [[100.0, 10.0]]
        assert d["asks"] == [[100.1, 5.0]]
        assert d["sequence"] == 12345


class TestTrade:
    """Test Trade dataclass."""
    
    def test_creation(self):
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            trade_id="12345",
            price=100.0,
            quantity=1.5,
            is_buyer_maker=True,
        )
        
        assert trade.exchange == "binance"
        assert trade.price == 100.0
        assert trade.quantity == 1.5
    
    def test_side(self):
        buy_trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            trade_id="12345",
            price=100.0,
            quantity=1.5,
            is_buyer_maker=False,  # Buyer is taker = buy
        )
        
        sell_trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            trade_id="12346",
            price=100.0,
            quantity=1.5,
            is_buyer_maker=True,  # Buyer is maker = sell
        )
        
        assert buy_trade.side == "buy"
        assert sell_trade.side == "sell"
    
    def test_to_dict(self):
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            timestamp=1000,
            local_timestamp=1001,
            trade_id="12345",
            price=100.0,
            quantity=1.5,
            is_buyer_maker=True,
        )
        
        d = trade.to_dict()
        
        assert d["exchange"] == "binance"
        assert d["price"] == 100.0
        assert d["side"] == "sell"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
