"""
Binance WebSocket data ingestion for Level 2 Order Book and Trades.

Uses Binance's combined streams for efficient data collection:
- Depth stream: Partial book depth (top N levels)
- Trade stream: Individual trades (aggTrade for aggregated trades)
"""
import json
import time
import asyncio
import logging
from typing import List, Optional, Callable

import websockets

from .base import DataIngestionBase, OrderBookSnapshot, Trade, PriceLevel

logger = logging.getLogger(__name__)


class BinanceIngestion(DataIngestionBase):
    """
    Binance WebSocket data ingestion.
    
    Subscribes to:
    - depth{N}@100ms: Top N price levels, updated every 100ms
    - aggTrade: Aggregated trades stream
    """
    
    WS_BASE_URL = "wss://stream.binance.com:9443/stream"
    WS_FUTURES_URL = "wss://fstream.binance.com/stream"
    
    def __init__(
        self,
        symbols: List[str],
        orderbook_depth: int = 20,
        on_orderbook: Optional[Callable[[OrderBookSnapshot], None]] = None,
        on_trade: Optional[Callable[[Trade], None]] = None,
        use_futures: bool = True,  # Use futures for better depth/liquidity
    ):
        super().__init__(symbols, orderbook_depth, on_orderbook, on_trade)
        self.use_futures = use_futures
        self._ws_url = self.WS_FUTURES_URL if use_futures else self.WS_BASE_URL
        
    @property
    def exchange_name(self) -> str:
        return "binance_futures" if self.use_futures else "binance"
    
    def _get_depth_level(self) -> int:
        """Get the nearest valid depth level for Binance."""
        # Binance supports: 5, 10, 20 for partial depth
        if self.orderbook_depth <= 5:
            return 5
        elif self.orderbook_depth <= 10:
            return 10
        else:
            return 20
    
    def _build_stream_names(self) -> List[str]:
        """Build the list of stream names to subscribe to."""
        streams = []
        depth_level = self._get_depth_level()
        
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            # Depth stream: depth{levels}@100ms for frequent updates
            streams.append(f"{symbol_lower}@depth{depth_level}@100ms")
            # Aggregated trades stream
            streams.append(f"{symbol_lower}@aggTrade")
        
        return streams
    
    async def connect(self):
        """Establish WebSocket connection to Binance."""
        streams = self._build_stream_names()
        stream_param = "/".join(streams)
        url = f"{self._ws_url}?streams={stream_param}"
        
        logger.info(f"Connecting to Binance: {url[:100]}...")
        self._ws = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message size
        )
        logger.info("Connected to Binance WebSocket")
    
    async def subscribe(self):
        """Subscribe to streams (already subscribed via URL params)."""
        # Binance combined streams subscribe via URL, no additional subscription needed
        pass
    
    async def _listen(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self._ws:
                if not self._running:
                    break
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Binance WebSocket closed: {e}")
            raise
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        local_ts = int(time.time() * 1000)
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            return
        
        # Binance combined stream format: {"stream": "...", "data": {...}}
        if "stream" not in data or "data" not in data:
            return
        
        stream_name = data["stream"]
        payload = data["data"]
        
        if "@depth" in stream_name:
            await self._handle_depth(payload, local_ts)
        elif "@aggTrade" in stream_name:
            await self._handle_trade(payload, local_ts)
    
    async def _handle_depth(self, data: dict, local_ts: int):
        """Handle depth (order book) update."""
        # Futures format differs slightly from spot
        symbol = data.get("s", "").upper()
        if not symbol:
            return
        
        # Parse bids and asks
        bids = [
            PriceLevel(price=float(level[0]), quantity=float(level[1]))
            for level in data.get("b", data.get("bids", []))
        ]
        asks = [
            PriceLevel(price=float(level[0]), quantity=float(level[1]))
            for level in data.get("a", data.get("asks", []))
        ]
        
        # Trim to requested depth
        bids = bids[:self.orderbook_depth]
        asks = asks[:self.orderbook_depth]
        
        snapshot = OrderBookSnapshot(
            exchange=self.exchange_name,
            symbol=symbol,
            timestamp=data.get("E", data.get("T", local_ts)),  # Event time or Transaction time
            local_timestamp=local_ts,
            bids=bids,
            asks=asks,
            sequence=data.get("u"),  # Final update ID
        )
        
        self._orderbooks[symbol] = snapshot
        self._emit_orderbook(snapshot)
    
    async def _handle_trade(self, data: dict, local_ts: int):
        """Handle aggregated trade."""
        symbol = data.get("s", "").upper()
        if not symbol:
            return
        
        trade = Trade(
            exchange=self.exchange_name,
            symbol=symbol,
            timestamp=data.get("T", data.get("E", local_ts)),  # Trade time
            local_timestamp=local_ts,
            trade_id=str(data.get("a", data.get("t", ""))),  # Aggregate trade ID
            price=float(data.get("p", 0)),
            quantity=float(data.get("q", 0)),
            is_buyer_maker=data.get("m", False),  # True if buyer is maker
        )
        
        self._emit_trade(trade)


class BinanceBookTickerIngestion(DataIngestionBase):
    """
    Ultra-low latency ingestion using bookTicker stream.
    
    Only provides best bid/ask (not full depth), but with minimal latency.
    Useful for execution layer.
    """
    
    WS_FUTURES_URL = "wss://fstream.binance.com/stream"
    
    def __init__(
        self,
        symbols: List[str],
        on_orderbook: Optional[Callable[[OrderBookSnapshot], None]] = None,
    ):
        super().__init__(symbols, orderbook_depth=1, on_orderbook=on_orderbook)
    
    @property
    def exchange_name(self) -> str:
        return "binance_futures_ticker"
    
    async def connect(self):
        streams = [f"{s.lower()}@bookTicker" for s in self.symbols]
        stream_param = "/".join(streams)
        url = f"{self.WS_FUTURES_URL}?streams={stream_param}"
        
        self._ws = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info("Connected to Binance bookTicker stream")
    
    async def subscribe(self):
        pass
    
    async def _listen(self):
        try:
            async for message in self._ws:
                if not self._running:
                    break
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Binance bookTicker WebSocket closed: {e}")
            raise
    
    async def _handle_message(self, message: str):
        local_ts = int(time.time() * 1000)
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        
        if "stream" not in data or "data" not in data:
            return
        
        payload = data["data"]
        symbol = payload.get("s", "").upper()
        
        if not symbol:
            return
        
        snapshot = OrderBookSnapshot(
            exchange=self.exchange_name,
            symbol=symbol,
            timestamp=payload.get("E", payload.get("T", local_ts)),
            local_timestamp=local_ts,
            bids=[PriceLevel(float(payload["b"]), float(payload["B"]))],
            asks=[PriceLevel(float(payload["a"]), float(payload["A"]))],
            sequence=payload.get("u"),
        )
        
        self._orderbooks[symbol] = snapshot
        self._emit_orderbook(snapshot)
