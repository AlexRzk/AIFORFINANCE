"""
Real data loader for backtesting - fetches actual market data from Binance/Bybit.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BinanceDataLoader:
    """Fetch OHLCV data from Binance."""
    
    @staticmethod
    def fetch_data(
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        days: int = 365,
        use_futures: bool = True,
    ) -> np.ndarray:
        """Fetch historical data from Binance.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Candle interval (1m, 5m, 1h, 4h, 1d)
            days: Number of days of historical data
            use_futures: Use futures or spot market
            
        Returns:
            Array of close prices
        """
        try:
            import ccxt
        except ImportError:
            logger.warning("ccxt not installed, using synthetic data")
            return None
        
        try:
            if use_futures:
                exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
            else:
                exchange = ccxt.binance({'enableRateLimit': True})
            
            logger.info(f"Fetching {days} days of {symbol} data from Binance...")
            
            # Convert interval to milliseconds
            interval_ms = {
                '1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000,
                '4h': 14400000, '1d': 86400000
            }.get(interval, 3600000)
            
            all_ohlcv = []
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (days * 24 * 3600 * 1000)
            
            current_time = start_time
            while current_time < end_time:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, interval, current_time)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    current_time = ohlcv[-1][0] + interval_ms
                    logger.info(f"  Fetched up to {datetime.fromtimestamp(current_time/1000)}")
                except Exception as e:
                    logger.warning(f"Error fetching batch: {e}")
                    break
            
            if not all_ohlcv:
                logger.warning("No data fetched, using synthetic data")
                return None
            
            # Extract close prices
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            logger.info(f"✅ Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df['close'].values.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}")
            return None


class LocalDataLoader:
    """Load data from local Parquet files (from Phase 1 data recording)."""
    
    @staticmethod
    def load_from_parquet(
        file_path: str,
        limit: Optional[int] = None,
    ) -> np.ndarray:
        """Load price data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            limit: Max number of rows to load
            
        Returns:
            Array of close prices
        """
        try:
            df = pd.read_parquet(file_path)
            
            if 'price' in df.columns:
                prices = df['price'].values
            elif 'close' in df.columns:
                prices = df['close'].values
            else:
                raise ValueError(f"Could not find price column in {df.columns}")
            
            if limit:
                prices = prices[:limit]
            
            logger.info(f"✅ Loaded {len(prices)} prices from {file_path}")
            
            return prices.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading Parquet file: {e}")
            return None


def get_price_data(
    data_source: str = "synthetic",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 365,
    n_synthetic: int = 100000,
) -> np.ndarray:
    """Get price data from various sources.
    
    Args:
        data_source: "synthetic", "binance", "bybit", or path to parquet
        symbol: Trading pair
        interval: Candle interval
        days: Days of history
        n_synthetic: Length of synthetic data
        
    Returns:
        Array of close prices
    """
    
    if data_source == "synthetic":
        logger.info(f"Generating {n_synthetic} synthetic price points...")
        return generate_synthetic_data(n_synthetic)
    
    elif data_source == "binance":
        prices = BinanceDataLoader.fetch_data(symbol, interval, days)
        if prices is None:
            logger.info("Falling back to synthetic data")
            return generate_synthetic_data(n_synthetic)
        return prices
    
    elif data_source == "bybit":
        # Bybit support can be added similarly
        logger.warning("Bybit support not yet implemented")
        return generate_synthetic_data(n_synthetic)
    
    elif ".parquet" in data_source:
        prices = LocalDataLoader.load_from_parquet(data_source)
        if prices is None:
            logger.info("Falling back to synthetic data")
            return generate_synthetic_data(n_synthetic)
        return prices
    
    else:
        logger.warning(f"Unknown data source: {data_source}, using synthetic")
        return generate_synthetic_data(n_synthetic)


def generate_synthetic_data(n_samples: int = 100000) -> np.ndarray:
    """Generate realistic GARCH-like price data.
    
    Mimics crypto market characteristics:
    - Volatility clustering
    - Fat tails
    - Mean reversion
    """
    np.random.seed(42)
    
    # GARCH(1,1) volatility
    volatility = np.zeros(n_samples)
    volatility[0] = 0.001
    
    for i in range(1, n_samples):
        # σ_t = sqrt(0.00001 + 0.1*σ_{t-1} + 0.85*r_{t-1}^2)
        volatility[i] = np.sqrt(
            0.00001 + 0.1 * volatility[i-1]**2 + 0.85 * (np.random.randn()**2) * 0.0001
        )
    
    # Generate returns
    returns = np.random.randn(n_samples) * volatility + 0.00001
    
    # Add mean reversion component
    mean_level = np.zeros(n_samples)
    for i in range(1, n_samples):
        mean_level[i] = 0.95 * mean_level[i-1] + 0.05 * returns[i-1]
    
    returns += 0.1 * mean_level
    
    # Generate prices starting at 50000
    log_prices = np.cumsum(returns) + np.log(50000)
    prices = np.exp(log_prices)
    
    return prices.astype(np.float32)


def split_data(
    prices: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test.
    
    Args:
        prices: Full price array
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        
    Returns:
        Tuple of (train_prices, val_prices, test_prices)
    """
    n = len(prices)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train = prices[:train_idx]
    val = prices[train_idx:val_idx]
    test = prices[val_idx:]
    
    logger.info(f"Data split: train={len(train)} | val={len(val)} | test={len(test)}")
    
    return train, val, test


def get_statistics(prices: np.ndarray) -> dict:
    """Compute statistics on price data."""
    returns = np.diff(np.log(prices))
    
    return {
        "n_candles": len(prices),
        "start_price": prices[0],
        "end_price": prices[-1],
        "total_return": (prices[-1] - prices[0]) / prices[0],
        "mean_return": np.mean(returns),
        "volatility": np.std(returns),
        "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(24 * 252),
        "max_price": np.max(prices),
        "min_price": np.min(prices),
    }
