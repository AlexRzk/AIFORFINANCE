"""
Base classes and data structures for data ingestion.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """Single price level in the order book."""
    price: float
    quantity: float


@dataclass
class OrderBookSnapshot:
    """
    Snapshot of the Limit Order Book at a specific timestamp.
    
    Attributes:
        exchange: Exchange name (binance, bybit)
        symbol: Trading pair (e.g., BTCUSDT)
        timestamp: Unix timestamp in milliseconds
        local_timestamp: Local reception time in milliseconds
        bids: List of bid price levels (sorted by price descending)
        asks: List of ask price levels (sorted by price ascending)
        sequence: Exchange sequence number for ordering
    """
    exchange: str
    symbol: str
    timestamp: int  # Exchange timestamp in ms
    local_timestamp: int  # Local reception time in ms
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    sequence: Optional[int] = None
    
    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Get the best (highest) bid."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Get the best (lowest) ask."""
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "local_timestamp": self.local_timestamp,
            "sequence": self.sequence,
            "bids": [[l.price, l.quantity] for l in self.bids],
            "asks": [[l.price, l.quantity] for l in self.asks],
        }


@dataclass
class Trade:
    """
    A single trade execution.
    
    Attributes:
        exchange: Exchange name
        symbol: Trading pair
        timestamp: Trade execution time in ms
        local_timestamp: Local reception time in ms
        trade_id: Exchange trade ID
        price: Trade price
        quantity: Trade quantity
        is_buyer_maker: True if the buyer was the maker (sell aggressor)
    """
    exchange: str
    symbol: str
    timestamp: int
    local_timestamp: int
    trade_id: str
    price: float
    quantity: float
    is_buyer_maker: bool  # True = sell aggressor, False = buy aggressor
    
    @property
    def side(self) -> str:
        """Get the aggressor side."""
        return "sell" if self.is_buyer_maker else "buy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "local_timestamp": self.local_timestamp,
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "is_buyer_maker": self.is_buyer_maker,
            "side": self.side,
        }


class DataIngestionBase(ABC):
    """
    Abstract base class for exchange data ingestion.
    
    Subclasses implement exchange-specific WebSocket handling.
    """
    
    def __init__(
        self,
        symbols: List[str],
        orderbook_depth: int = 20,
        on_orderbook: Optional[Callable[[OrderBookSnapshot], None]] = None,
        on_trade: Optional[Callable[[Trade], None]] = None,
    ):
        """
        Initialize the data ingestion.
        
        Args:
            symbols: List of trading pairs to subscribe to
            orderbook_depth: Number of price levels to maintain
            on_orderbook: Callback for orderbook updates
            on_trade: Callback for trade updates
        """
        self.symbols = [s.upper() for s in symbols]
        self.orderbook_depth = orderbook_depth
        self.on_orderbook = on_orderbook
        self.on_trade = on_trade
        
        self._running = False
        self._ws = None
        self._orderbooks: Dict[str, OrderBookSnapshot] = {}
        
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Return the exchange name."""
        pass
    
    @abstractmethod
    async def connect(self):
        """Establish WebSocket connection."""
        pass
    
    @abstractmethod
    async def subscribe(self):
        """Subscribe to orderbook and trade streams."""
        pass
    
    @abstractmethod
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        pass
    
    async def start(self):
        """Start the data ingestion loop."""
        self._running = True
        logger.info(f"Starting {self.exchange_name} data ingestion for {self.symbols}")
        
        while self._running:
            try:
                await self.connect()
                await self.subscribe()
                await self._listen()
            except Exception as e:
                logger.error(f"Error in {self.exchange_name} ingestion: {e}")
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the data ingestion."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info(f"Stopped {self.exchange_name} data ingestion")
    
    async def _listen(self):
        """Listen for WebSocket messages."""
        async for message in self._ws:
            if not self._running:
                break
            await self._handle_message(message)
    
    def _emit_orderbook(self, snapshot: OrderBookSnapshot):
        """Emit orderbook snapshot to callback."""
        if self.on_orderbook:
            try:
                self.on_orderbook(snapshot)
            except Exception as e:
                logger.error(f"Error in orderbook callback: {e}")
    
    def _emit_trade(self, trade: Trade):
        """Emit trade to callback."""
        if self.on_trade:
            try:
                self.on_trade(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    def get_current_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get the current orderbook for a symbol."""
        return self._orderbooks.get(symbol.upper())
