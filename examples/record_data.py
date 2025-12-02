"""
Example: Start Live Data Recording

This script connects to Binance Futures WebSocket and records
LOB snapshots and trades to Parquet files.

Usage:
    python examples/record_data.py --symbols BTCUSDT ETHUSDT --duration 3600
"""
import asyncio
import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import BinanceIngestion, BybitIngestion, DataRecorder
from src.config import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main(
    symbols: list,
    exchange: str = "binance",
    duration_seconds: int = None,
    data_dir: str = "./data",
):
    """
    Main data recording function.
    
    Args:
        symbols: List of trading pairs to record
        exchange: Exchange to connect to (binance/bybit)
        duration_seconds: Recording duration (None = indefinite)
        data_dir: Directory for data storage
    """
    # Initialize recorder
    recorder = DataRecorder(
        base_dir=Path(data_dir),
        flush_interval_seconds=10.0,
        max_buffer_size=10000,
    )
    
    # Initialize exchange connection
    if exchange == "binance":
        ingestion = BinanceIngestion(
            symbols=symbols,
            orderbook_depth=20,
            on_orderbook=recorder.record_orderbook,
            on_trade=recorder.record_trade,
            use_futures=True,
        )
    elif exchange == "bybit":
        ingestion = BybitIngestion(
            symbols=symbols,
            orderbook_depth=20,
            on_orderbook=recorder.record_orderbook,
            on_trade=recorder.record_trade,
            market_type="linear",
        )
    else:
        raise ValueError(f"Unknown exchange: {exchange}")
    
    logger.info(f"Starting data recording for {symbols} on {exchange}")
    logger.info(f"Data will be stored in: {data_dir}")
    
    try:
        # Start recorder and ingestion
        await recorder.start()
        
        if duration_seconds:
            # Record for specified duration
            ingestion_task = asyncio.create_task(ingestion.start())
            await asyncio.sleep(duration_seconds)
            await ingestion.stop()
            ingestion_task.cancel()
        else:
            # Record indefinitely
            await ingestion.start()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping...")
    finally:
        await ingestion.stop()
        await recorder.stop()
        
        # Print statistics
        stats = recorder.get_stats()
        logger.info(f"Recording complete: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record market data from exchanges")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT"],
        help="Trading pairs to record"
    )
    parser.add_argument(
        "--exchange",
        choices=["binance", "bybit"],
        default="binance",
        help="Exchange to connect to"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Recording duration in seconds (default: indefinite)"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory for data storage"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(
        symbols=args.symbols,
        exchange=args.exchange,
        duration_seconds=args.duration,
        data_dir=args.data_dir,
    ))
