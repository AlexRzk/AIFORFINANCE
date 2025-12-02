"""
Bybit WebSocket data ingestion for Level 2 Order Book and Trades.

Uses Bybit's V5 API for unified access to derivatives markets.
"""
import json
import time
import asyncio
import logging
from typing import List, Optional, Callable, Dict

import websockets

from .base import DataIngestionBase, OrderBookSnapshot, Trade, PriceLevel

logger = logging.getLogger(__name__)


class BybitIngestion(DataIngestionBase):
    """
    Bybit WebSocket data ingestion (V5 API).
    
    Subscribes to:
    - orderbook.{depth}.{symbol}: Order book snapshots/deltas
    - publicTrade.{symbol}: Public trades
    """
    
    WS_LINEAR_URL = "wss://stream.bybit.com/v5/public/linear"  # USDT perpetual
    WS_INVERSE_URL = "wss://stream.bybit.com/v5/public/inverse"  # Inverse perpetual
    WS_SPOT_URL = "wss://stream.bybit.com/v5/public/spot"
    
    def __init__(
        self,
        symbols: List[str],
        orderbook_depth: int = 20,
        on_orderbook: Optional[Callable[[OrderBookSnapshot], None]] = None,
        on_trade: Optional[Callable[[Trade], None]] = None,
        market_type: str = "linear",  # linear, inverse, spot
    ):
        super().__init__(symbols, orderbook_depth, on_orderbook, on_trade)
        self.market_type = market_type
        
        if market_type == "linear":
            self._ws_url = self.WS_LINEAR_URL
        elif market_type == "inverse":
            self._ws_url = self.WS_INVERSE_URL
        else:
            self._ws_url = self.WS_SPOT_URL
        
        # Bybit sends snapshots and then deltas; we maintain local book state
        self._local_books: Dict[str, Dict] = {}
        
    @property
    def exchange_name(self) -> str:
        return f"bybit_{self.market_type}"
    
    def _get_depth_level(self) -> int:
        """Get the nearest valid depth level for Bybit."""
        # Bybit supports: 1, 50, 200, 500 for orderbook
        # For low latency, use 50
        if self.orderbook_depth <= 1:
            return 1
        elif self.orderbook_depth <= 50:
            return 50
        elif self.orderbook_depth <= 200:
            return 200
        else:
            return 500
    
    async def connect(self):
        """Establish WebSocket connection to Bybit."""
        logger.info(f"Connecting to Bybit: {self._ws_url}")
        self._ws = await websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,
        )
        logger.info("Connected to Bybit WebSocket")
    
    async def subscribe(self):
        """Subscribe to orderbook and trade streams."""
        depth_level = self._get_depth_level()
        
        topics = []
        for symbol in self.symbols:
            # Order book topic
            topics.append(f"orderbook.{depth_level}.{symbol}")
            # Public trades topic
            topics.append(f"publicTrade.{symbol}")
        
        subscribe_msg = {
            "op": "subscribe",
            "args": topics
        }
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to Bybit topics: {topics}")
    
    async def _listen(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self._ws:
                if not self._running:
                    break
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Bybit WebSocket closed: {e}")
            raise
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        local_ts = int(time.time() * 1000)
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            return
        
        # Handle subscription confirmation
        if data.get("op") == "subscribe":
            if data.get("success"):
                logger.info("Bybit subscription confirmed")
            else:
                logger.error(f"Bybit subscription failed: {data}")
            return
        
        # Handle pong
        if data.get("op") == "pong":
            return
        
        topic = data.get("topic", "")
        
        if topic.startswith("orderbook."):
            await self._handle_orderbook(data, local_ts)
        elif topic.startswith("publicTrade."):
            await self._handle_trades(data, local_ts)
    
    async def _handle_orderbook(self, data: dict, local_ts: int):
        """Handle orderbook update (snapshot or delta)."""
        topic = data.get("topic", "")
        msg_type = data.get("type", "")  # snapshot or delta
        payload = data.get("data", {})
        
        # Extract symbol from topic: orderbook.50.BTCUSDT -> BTCUSDT
        parts = topic.split(".")
        if len(parts) < 3:
            return
        symbol = parts[2].upper()
        
        if msg_type == "snapshot":
            # Full snapshot - replace local book
            self._local_books[symbol] = {
                "bids": {float(level[0]): float(level[1]) for level in payload.get("b", [])},
                "asks": {float(level[0]): float(level[1]) for level in payload.get("a", [])},
            }
        elif msg_type == "delta":
            # Delta update - apply changes
            if symbol not in self._local_books:
                logger.warning(f"Delta received before snapshot for {symbol}")
                return
            
            book = self._local_books[symbol]
            
            # Apply bid updates
            for level in payload.get("b", []):
                price, qty = float(level[0]), float(level[1])
                if qty == 0:
                    book["bids"].pop(price, None)
                else:
                    book["bids"][price] = qty
            
            # Apply ask updates
            for level in payload.get("a", []):
                price, qty = float(level[0]), float(level[1])
                if qty == 0:
                    book["asks"].pop(price, None)
                else:
                    book["asks"][price] = qty
        
        # Build snapshot from local book
        if symbol in self._local_books:
            book = self._local_books[symbol]
            
            # Sort and limit depth
            bids = sorted(book["bids"].items(), key=lambda x: -x[0])[:self.orderbook_depth]
            asks = sorted(book["asks"].items(), key=lambda x: x[0])[:self.orderbook_depth]
            
            snapshot = OrderBookSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                timestamp=data.get("ts", local_ts),
                local_timestamp=local_ts,
                bids=[PriceLevel(price=p, quantity=q) for p, q in bids],
                asks=[PriceLevel(price=p, quantity=q) for p, q in asks],
                sequence=payload.get("u"),  # Update ID
            )
            
            self._orderbooks[symbol] = snapshot
            self._emit_orderbook(snapshot)
    
    async def _handle_trades(self, data: dict, local_ts: int):
        """Handle public trades."""
        payload = data.get("data", [])
        
        if not isinstance(payload, list):
            payload = [payload]
        
        for trade_data in payload:
            symbol = trade_data.get("s", "").upper()
            if not symbol:
                continue
            
            # Bybit uses "S" for side: Buy or Sell
            side = trade_data.get("S", "").lower()
            is_buyer_maker = side == "sell"  # Sell = buyer is maker
            
            trade = Trade(
                exchange=self.exchange_name,
                symbol=symbol,
                timestamp=trade_data.get("T", local_ts),  # Trade timestamp
                local_timestamp=local_ts,
                trade_id=str(trade_data.get("i", "")),  # Trade ID
                price=float(trade_data.get("p", 0)),
                quantity=float(trade_data.get("v", 0)),  # Volume
                is_buyer_maker=is_buyer_maker,
            )
            
            self._emit_trade(trade)
    
    async def _send_ping(self):
        """Send heartbeat ping."""
        while self._running:
            try:
                await asyncio.sleep(20)
                if self._ws and not self._ws.closed:
                    await self._ws.send(json.dumps({"op": "ping"}))
            except Exception as e:
                logger.error(f"Ping error: {e}")
    
    async def start(self):
        """Start with ping task."""
        self._running = True
        
        while self._running:
            try:
                await self.connect()
                await self.subscribe()
                
                # Start ping task
                ping_task = asyncio.create_task(self._send_ping())
                
                try:
                    await self._listen()
                finally:
                    ping_task.cancel()
                    
            except Exception as e:
                logger.error(f"Error in Bybit ingestion: {e}")
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
