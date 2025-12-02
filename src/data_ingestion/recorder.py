"""
Data Recorder - Stores LOB snapshots and trades to Parquet files.

Provides efficient batched writing with configurable flush intervals.
Uses PyArrow for high-performance Parquet I/O.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq

from .base import OrderBookSnapshot, Trade

logger = logging.getLogger(__name__)


class DataRecorder:
    """
    High-performance data recorder for LOB and trade data.
    
    Features:
    - Batched writes to reduce I/O overhead
    - Separate files per symbol and date
    - Snappy compression for efficient storage
    - Thread-safe buffer management
    """
    
    def __init__(
        self,
        base_dir: Path,
        flush_interval_seconds: float = 10.0,
        max_buffer_size: int = 10000,
        compression: str = "snappy",
    ):
        """
        Initialize the data recorder.
        
        Args:
            base_dir: Base directory for data storage
            flush_interval_seconds: Time between automatic flushes
            max_buffer_size: Maximum records before forced flush
            compression: Parquet compression (snappy, gzip, zstd)
        """
        self.base_dir = Path(base_dir)
        self.flush_interval = flush_interval_seconds
        self.max_buffer_size = max_buffer_size
        self.compression = compression
        
        # Create directories
        self.lob_dir = self.base_dir / "lob"
        self.trades_dir = self.base_dir / "trades"
        self.lob_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffers for batched writes
        self._lob_buffer: List[Dict[str, Any]] = []
        self._trades_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = Lock()
        
        # Statistics
        self._lob_count = 0
        self._trade_count = 0
        self._last_flush = time.time()
        
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        
    def record_orderbook(self, snapshot: OrderBookSnapshot):
        """Record an orderbook snapshot."""
        # Flatten the snapshot for columnar storage
        record = {
            "exchange": snapshot.exchange,
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp,
            "local_timestamp": snapshot.local_timestamp,
            "sequence": snapshot.sequence,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
        }
        
        # Add price levels as separate columns
        for i, bid in enumerate(snapshot.bids[:20]):
            record[f"bid_price_{i}"] = bid.price
            record[f"bid_qty_{i}"] = bid.quantity
        
        for i, ask in enumerate(snapshot.asks[:20]):
            record[f"ask_price_{i}"] = ask.price
            record[f"ask_qty_{i}"] = ask.quantity
        
        with self._buffer_lock:
            self._lob_buffer.append(record)
            self._lob_count += 1
            
            if len(self._lob_buffer) >= self.max_buffer_size:
                self._flush_lob_buffer()
    
    def record_trade(self, trade: Trade):
        """Record a trade."""
        record = trade.to_dict()
        
        with self._buffer_lock:
            self._trades_buffer.append(record)
            self._trade_count += 1
            
            if len(self._trades_buffer) >= self.max_buffer_size:
                self._flush_trades_buffer()
    
    def _get_file_path(self, base_dir: Path, exchange: str, symbol: str, suffix: str = "") -> Path:
        """Get the file path for a given exchange, symbol, and date."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{exchange}_{symbol}_{date_str}{suffix}.parquet"
        return base_dir / exchange / symbol / filename
    
    def _flush_lob_buffer(self):
        """Flush LOB buffer to disk."""
        if not self._lob_buffer:
            return
        
        # Group by exchange and symbol
        grouped: Dict[tuple, List[Dict]] = {}
        for record in self._lob_buffer:
            key = (record["exchange"], record["symbol"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        
        self._lob_buffer = []
        
        for (exchange, symbol), records in grouped.items():
            try:
                self._write_parquet(
                    self._get_file_path(self.lob_dir, exchange, symbol),
                    records,
                    self._get_lob_schema()
                )
            except Exception as e:
                logger.error(f"Failed to write LOB data: {e}")
    
    def _flush_trades_buffer(self):
        """Flush trades buffer to disk."""
        if not self._trades_buffer:
            return
        
        # Group by exchange and symbol
        grouped: Dict[tuple, List[Dict]] = {}
        for record in self._trades_buffer:
            key = (record["exchange"], record["symbol"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        
        self._trades_buffer = []
        
        for (exchange, symbol), records in grouped.items():
            try:
                self._write_parquet(
                    self._get_file_path(self.trades_dir, exchange, symbol),
                    records,
                    self._get_trades_schema()
                )
            except Exception as e:
                logger.error(f"Failed to write trades data: {e}")
    
    def _write_parquet(self, path: Path, records: List[Dict], schema: pa.Schema):
        """Write records to a Parquet file (append if exists)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(records, schema=schema)
        
        if path.exists():
            # Append to existing file
            existing_table = pq.read_table(path)
            table = pa.concat_tables([existing_table, table])
        
        # Write with compression
        pq.write_table(
            table,
            path,
            compression=self.compression,
            use_dictionary=True,
        )
        
        logger.debug(f"Wrote {len(records)} records to {path}")
    
    def _get_lob_schema(self) -> pa.Schema:
        """Get PyArrow schema for LOB data."""
        fields = [
            ("exchange", pa.string()),
            ("symbol", pa.string()),
            ("timestamp", pa.int64()),
            ("local_timestamp", pa.int64()),
            ("sequence", pa.int64()),
            ("mid_price", pa.float64()),
            ("spread", pa.float64()),
        ]
        
        # Add 20 levels for bids and asks
        for i in range(20):
            fields.append((f"bid_price_{i}", pa.float64()))
            fields.append((f"bid_qty_{i}", pa.float64()))
            fields.append((f"ask_price_{i}", pa.float64()))
            fields.append((f"ask_qty_{i}", pa.float64()))
        
        return pa.schema(fields)
    
    def _get_trades_schema(self) -> pa.Schema:
        """Get PyArrow schema for trade data."""
        return pa.schema([
            ("exchange", pa.string()),
            ("symbol", pa.string()),
            ("timestamp", pa.int64()),
            ("local_timestamp", pa.int64()),
            ("trade_id", pa.string()),
            ("price", pa.float64()),
            ("quantity", pa.float64()),
            ("is_buyer_maker", pa.bool_()),
            ("side", pa.string()),
        ])
    
    async def _periodic_flush(self):
        """Periodically flush buffers to disk."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            self.flush()
    
    def flush(self):
        """Force flush all buffers to disk."""
        with self._buffer_lock:
            self._flush_lob_buffer()
            self._flush_trades_buffer()
        
        self._last_flush = time.time()
        logger.info(
            f"Flushed data: {self._lob_count} LOB snapshots, "
            f"{self._trade_count} trades"
        )
    
    async def start(self):
        """Start the periodic flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"DataRecorder started, writing to {self.base_dir}")
    
    async def stop(self):
        """Stop the recorder and flush remaining data."""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        self.flush()
        logger.info("DataRecorder stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        return {
            "lob_count": self._lob_count,
            "trade_count": self._trade_count,
            "lob_buffer_size": len(self._lob_buffer),
            "trades_buffer_size": len(self._trades_buffer),
            "last_flush": self._last_flush,
        }


class StreamingParquetWriter:
    """
    Memory-efficient streaming Parquet writer using row groups.
    
    For very high-frequency data where batched writes are insufficient.
    """
    
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        row_group_size: int = 10000,
        compression: str = "snappy",
    ):
        self.path = Path(path)
        self.schema = schema
        self.row_group_size = row_group_size
        self.compression = compression
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self._buffer: List[Dict] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._rows_written = 0
    
    def write(self, record: Dict):
        """Write a single record."""
        self._buffer.append(record)
        
        if len(self._buffer) >= self.row_group_size:
            self._flush()
    
    def _flush(self):
        """Flush buffer as a row group."""
        if not self._buffer:
            return
        
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression=self.compression,
            )
        
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._buffer = []
    
    def close(self):
        """Close the writer."""
        self._flush()
        if self._writer:
            self._writer.close()
            self._writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
