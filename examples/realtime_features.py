"""
Example: Real-time Feature Engineering

This script demonstrates real-time feature computation from
WebSocket data, including OFI and fractional differentiation.

Usage:
    python examples/realtime_features.py --symbol BTCUSDT --duration 60
"""
import asyncio
import argparse
import logging
from pathlib import Path
import sys
import time
from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import BinanceIngestion, OrderBookSnapshot, Trade
from src.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RealTimeFeatureProcessor:
    """
    Real-time feature processor that computes features from
    streaming market data.
    """
    
    def __init__(self, symbol: str, d_price: float = 0.4):
        self.symbol = symbol
        self.feature_engineer = FeatureEngineer(
            symbol=symbol,
            d_price=d_price,
            d_volume=0.5,
        )
        
        # Feature history for analysis
        self.feature_history: Deque[dict] = deque(maxlen=1000)
        self.update_count = 0
        
        # Timing
        self.last_print = time.time()
        self.print_interval = 5.0  # Print every 5 seconds
    
    def on_orderbook(self, snapshot: OrderBookSnapshot):
        """Process orderbook update."""
        if snapshot.symbol != self.symbol:
            return
        
        # Extract prices and quantities
        bid_prices = [level.price for level in snapshot.bids]
        bid_quantities = [level.quantity for level in snapshot.bids]
        ask_prices = [level.price for level in snapshot.asks]
        ask_quantities = [level.quantity for level in snapshot.asks]
        
        # Calculate mid price
        mid_price = snapshot.mid_price
        if mid_price is None:
            return
        
        # Estimate recent volume (simplified)
        volume = sum(bid_quantities[:5]) + sum(ask_quantities[:5])
        
        # Compute features
        features = self.feature_engineer.update(
            timestamp=snapshot.local_timestamp,
            mid_price=mid_price,
            volume=volume,
            bid_prices=bid_prices,
            bid_quantities=bid_quantities,
            ask_prices=ask_prices,
            ask_quantities=ask_quantities,
        )
        
        self.update_count += 1
        
        if features is not None:
            self.feature_history.append(features.to_dict())
            
            # Periodic logging
            now = time.time()
            if now - self.last_print > self.print_interval:
                self._print_status(features, snapshot)
                self.last_print = now
    
    def on_trade(self, trade: Trade):
        """Process trade update."""
        if trade.symbol != self.symbol:
            return
        
        self.feature_engineer.add_trade(
            timestamp=trade.local_timestamp,
            quantity=trade.quantity,
            is_buyer_maker=trade.is_buyer_maker,
        )
    
    def _print_status(self, features, snapshot):
        """Print current status."""
        logger.info(
            f"\n{'='*60}\n"
            f"Symbol: {self.symbol} | Updates: {self.update_count}\n"
            f"Mid Price: ${features.mid_price:.2f} | Spread: ${features.spread:.4f}\n"
            f"{'='*60}\n"
            f"FEATURES:\n"
            f"  Price FFD: {features.price_ffd:.6f}\n"
            f"  OFI (L0): {features.ofi_level_0:.2f} | OFI (1s): {features.ofi_1s:.2f}\n"
            f"  OFI Z-score: {features.ofi_zscore_1s:.2f}\n"
            f"  FDI: {features.fdi:.3f} ({'trending' if features.fdi < 1.5 else 'ranging'})\n"
            f"  Volatility: {features.volatility:.6f}\n"
            f"  Trade Imbalance: {features.trade_imbalance:.2f}\n"
            f"{'='*60}"
        )
    
    def get_feature_df(self) -> pd.DataFrame:
        """Get features as DataFrame."""
        return pd.DataFrame(list(self.feature_history))


async def main(symbol: str, duration: int = 60, d_price: float = 0.4):
    """
    Run real-time feature processing.
    
    Args:
        symbol: Trading pair symbol
        duration: Duration in seconds
        d_price: Fractional differentiation order for price
    """
    logger.info(f"Starting real-time feature processing for {symbol}")
    logger.info(f"Using d={d_price} for fractional differentiation")
    
    # Initialize processor
    processor = RealTimeFeatureProcessor(symbol=symbol, d_price=d_price)
    
    # Initialize exchange connection
    ingestion = BinanceIngestion(
        symbols=[symbol],
        orderbook_depth=20,
        on_orderbook=processor.on_orderbook,
        on_trade=processor.on_trade,
        use_futures=True,
    )
    
    try:
        # Start ingestion
        ingestion_task = asyncio.create_task(ingestion.start())
        
        # Wait for duration
        logger.info(f"Recording for {duration} seconds...")
        await asyncio.sleep(duration)
        
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await ingestion.stop()
    
    # Save features
    df = processor.get_feature_df()
    if len(df) > 0:
        output_path = Path("data/realtime_features.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} feature vectors to {output_path}")
        
        # Print summary statistics
        logger.info("\n=== Feature Statistics ===")
        for col in df.select_dtypes(include=[np.number]).columns:
            logger.info(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    else:
        logger.warning("No features computed (need more warm-up time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time feature processing")
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair symbol"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds"
    )
    parser.add_argument(
        "--d-price",
        type=float,
        default=0.4,
        help="Fractional differentiation order for price"
    )
    
    args = parser.parse_args()
    asyncio.run(main(
        symbol=args.symbol,
        duration=args.duration,
        d_price=args.d_price,
    ))
