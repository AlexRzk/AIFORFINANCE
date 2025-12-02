"""
Feature Engineering Module - Combines all features for ML model input.

This module orchestrates:
1. Order Flow Imbalance (OFI) features
2. Fractional Differentiation of price data
3. Additional technical indicators
4. Regime detection features (FDI)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .ofi import OrderFlowImbalance, OFIFeatures
from .fracdiff import FractionalDifferentiator, frac_diff_ffd, find_optimal_d

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Complete feature vector for a single timestep."""
    timestamp: int
    symbol: str
    
    # Price features (fractionally differentiated)
    price_ffd: float
    volume_ffd: float
    
    # OFI features
    ofi_level_0: float
    ofi_level_5: float
    ofi_1s: float
    ofi_5s: float
    ofi_zscore_1s: float
    
    # Microstructure features
    spread: float
    mid_price: float
    trade_imbalance: float
    
    # Regime features
    fdi: float  # Fractal Dimension Index
    volatility: float  # Recent volatility
    
    # External features (placeholders)
    global_risk_score: float = 0.5  # 0-1 scale
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.price_ffd,
            self.volume_ffd,
            self.ofi_level_0,
            self.ofi_level_5,
            self.ofi_1s,
            self.ofi_5s,
            self.ofi_zscore_1s,
            self.spread,
            self.trade_imbalance,
            self.fdi,
            self.volatility,
            self.global_risk_score,
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "price_ffd": self.price_ffd,
            "volume_ffd": self.volume_ffd,
            "ofi_level_0": self.ofi_level_0,
            "ofi_level_5": self.ofi_level_5,
            "ofi_1s": self.ofi_1s,
            "ofi_5s": self.ofi_5s,
            "ofi_zscore_1s": self.ofi_zscore_1s,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "trade_imbalance": self.trade_imbalance,
            "fdi": self.fdi,
            "volatility": self.volatility,
            "global_risk_score": self.global_risk_score,
        }
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get list of feature names."""
        return [
            "price_ffd", "volume_ffd", "ofi_level_0", "ofi_level_5",
            "ofi_1s", "ofi_5s", "ofi_zscore_1s", "spread",
            "trade_imbalance", "fdi", "volatility", "global_risk_score",
        ]


class FractalDimensionIndex:
    """
    Fractal Dimension Index (FDI) for regime detection.
    
    FDI measures whether the market is trending (FDI < 1.5) or
    ranging/mean-reverting (FDI > 1.5).
    
    Based on the Hurst exponent relationship:
    FDI = 1 + log(N) / log(Range/StdDev)
    
    A simpler approximation uses the ratio of the "rough" to "smooth"
    path lengths.
    """
    
    def __init__(self, window: int = 30):
        """
        Initialize FDI calculator.
        
        Args:
            window: Lookback window for calculation
        """
        self.window = window
        self._buffer: List[float] = []
    
    def update(self, price: float) -> Optional[float]:
        """
        Update with new price and calculate FDI.
        
        Args:
            price: New price observation
        
        Returns:
            FDI value (1.0 = trending, 2.0 = ranging)
        """
        self._buffer.append(price)
        
        if len(self._buffer) > self.window:
            self._buffer = self._buffer[-self.window:]
        
        if len(self._buffer) < self.window:
            return None
        
        return self._calculate_fdi()
    
    def _calculate_fdi(self) -> float:
        """Calculate FDI from buffer."""
        prices = np.array(self._buffer)
        n = len(prices)
        
        # Method 1: Box-counting dimension approximation
        # Normalize prices to [0, 1]
        price_range = prices.max() - prices.min()
        if price_range == 0:
            return 1.5  # Neutral
        
        normalized = (prices - prices.min()) / price_range
        
        # Calculate path length (sum of absolute changes)
        path_length = np.sum(np.abs(np.diff(normalized)))
        
        # Straight line distance (start to end)
        straight_dist = np.sqrt((1 / (n - 1)) ** 2 + (normalized[-1] - normalized[0]) ** 2)
        
        if straight_dist < 1e-10:
            straight_dist = 1e-10
        
        # FDI approximation
        fdi = 1 + np.log(path_length / straight_dist) / np.log(2 * (n - 1))
        
        # Clamp to valid range
        return np.clip(fdi, 1.0, 2.0)
    
    def reset(self):
        """Clear the buffer."""
        self._buffer = []


def calculate_fdi_series(prices: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate FDI for a price series.
    
    Args:
        prices: Price series
        window: Lookback window
    
    Returns:
        FDI series
    """
    fdi_values = []
    
    for i in range(len(prices)):
        if i < window - 1:
            fdi_values.append(np.nan)
        else:
            window_prices = prices.iloc[i - window + 1:i + 1].values
            
            # Normalize
            price_range = window_prices.max() - window_prices.min()
            if price_range == 0:
                fdi_values.append(1.5)
                continue
            
            normalized = (window_prices - window_prices.min()) / price_range
            
            # Path length
            path_length = np.sum(np.abs(np.diff(normalized)))
            straight_dist = np.sqrt(
                (1 / (window - 1)) ** 2 + (normalized[-1] - normalized[0]) ** 2
            )
            
            if straight_dist < 1e-10:
                straight_dist = 1e-10
            
            fdi = 1 + np.log(path_length / straight_dist) / np.log(2 * (window - 1))
            fdi_values.append(np.clip(fdi, 1.0, 2.0))
    
    return pd.Series(fdi_values, index=prices.index, name="fdi")


class FeatureEngineer:
    """
    Main feature engineering class for real-time feature computation.
    
    Combines:
    - OFI calculation
    - Fractional differentiation
    - Regime detection (FDI)
    - Technical indicators
    """
    
    def __init__(
        self,
        symbol: str,
        d_price: float = 0.4,
        d_volume: float = 0.5,
        ofi_timeframes_ms: List[int] = None,
        fdi_window: int = 30,
        volatility_window: int = 100,
    ):
        """
        Initialize the feature engineer.
        
        Args:
            symbol: Trading pair symbol
            d_price: Fractional differentiation order for price
            d_volume: Fractional differentiation order for volume
            ofi_timeframes_ms: OFI calculation timeframes
            fdi_window: Window for FDI calculation
            volatility_window: Window for volatility calculation
        """
        self.symbol = symbol
        
        # Fractional differentiators
        self.price_fracdiff = FractionalDifferentiator(d_price)
        self.volume_fracdiff = FractionalDifferentiator(d_volume)
        
        # OFI calculator
        self.ofi_calculator = OrderFlowImbalance(
            timeframes_ms=ofi_timeframes_ms or [100, 1000, 5000]
        )
        
        # FDI calculator
        self.fdi_calculator = FractalDimensionIndex(window=fdi_window)
        
        # Volatility buffer
        self.volatility_window = volatility_window
        self._returns_buffer: List[float] = []
        self._prev_price: Optional[float] = None
        
        # Previous orderbook state for differential OFI
        self._prev_bids: Optional[Tuple[List[float], List[float]]] = None
        self._prev_asks: Optional[Tuple[List[float], List[float]]] = None
    
    def update(
        self,
        timestamp: int,
        mid_price: float,
        volume: float,
        bid_prices: List[float],
        bid_quantities: List[float],
        ask_prices: List[float],
        ask_quantities: List[float],
        global_risk_score: float = 0.5,
    ) -> Optional[FeatureVector]:
        """
        Update with new market data and compute features.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            mid_price: Current mid price
            volume: Recent volume
            bid_prices: Bid price levels
            bid_quantities: Bid quantities
            ask_prices: Ask price levels
            ask_quantities: Ask quantities
            global_risk_score: External risk indicator (0-1)
        
        Returns:
            FeatureVector if enough data, None otherwise
        """
        # Log price for fractional differentiation
        log_price = np.log(mid_price) if mid_price > 0 else 0
        log_volume = np.log(volume + 1) if volume >= 0 else 0
        
        # Update fractional differentiators
        price_ffd = self.price_fracdiff.update(log_price)
        volume_ffd = self.volume_fracdiff.update(log_volume)
        
        # Update OFI
        ofi_features = self.ofi_calculator.update(
            symbol=self.symbol,
            timestamp=timestamp,
            bid_prices=bid_prices,
            bid_quantities=bid_quantities,
            ask_prices=ask_prices,
            ask_quantities=ask_quantities,
            prev_bid_prices=self._prev_bids[0] if self._prev_bids else None,
            prev_bid_quantities=self._prev_bids[1] if self._prev_bids else None,
            prev_ask_prices=self._prev_asks[0] if self._prev_asks else None,
            prev_ask_quantities=self._prev_asks[1] if self._prev_asks else None,
        )
        
        # Store current orderbook for next update
        self._prev_bids = (bid_prices.copy(), bid_quantities.copy())
        self._prev_asks = (ask_prices.copy(), ask_quantities.copy())
        
        # Update FDI
        fdi = self.fdi_calculator.update(mid_price)
        
        # Update volatility
        if self._prev_price is not None:
            ret = np.log(mid_price / self._prev_price)
            self._returns_buffer.append(ret)
            if len(self._returns_buffer) > self.volatility_window:
                self._returns_buffer = self._returns_buffer[-self.volatility_window:]
        self._prev_price = mid_price
        
        volatility = np.std(self._returns_buffer) if len(self._returns_buffer) > 1 else 0.0
        
        # Calculate spread
        spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0.0
        
        # Trade imbalance
        trade_imbalance = self.ofi_calculator.get_trade_imbalance(
            self.symbol, timestamp, window_ms=1000
        )
        
        # Skip if not enough data
        if price_ffd is None or fdi is None:
            return None
        
        return FeatureVector(
            timestamp=timestamp,
            symbol=self.symbol,
            price_ffd=price_ffd,
            volume_ffd=volume_ffd or 0.0,
            ofi_level_0=ofi_features.ofi_level_0,
            ofi_level_5=ofi_features.ofi_level_5,
            ofi_1s=ofi_features.ofi_1s,
            ofi_5s=ofi_features.ofi_5s,
            ofi_zscore_1s=ofi_features.ofi_zscore_1s,
            spread=spread,
            mid_price=mid_price,
            trade_imbalance=trade_imbalance,
            fdi=fdi,
            volatility=volatility,
            global_risk_score=global_risk_score,
        )
    
    def add_trade(
        self,
        timestamp: int,
        quantity: float,
        is_buyer_maker: bool,
    ):
        """Add a trade for imbalance calculation."""
        self.ofi_calculator.add_trade(
            self.symbol, timestamp, quantity, is_buyer_maker
        )
    
    def reset(self):
        """Reset all internal state."""
        self.price_fracdiff.reset()
        self.volume_fracdiff.reset()
        self.fdi_calculator.reset()
        self._returns_buffer = []
        self._prev_price = None
        self._prev_bids = None
        self._prev_asks = None


def engineer_features_from_dataframe(
    lob_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    d_price: float = None,
    d_volume: float = None,
    fdi_window: int = 30,
) -> pd.DataFrame:
    """
    Engineer features from LOB and trade DataFrames.
    
    Args:
        lob_df: LOB snapshots with columns:
            - timestamp, mid_price, spread
            - bid_price_0..N, bid_qty_0..N
            - ask_price_0..N, ask_qty_0..N
        trades_df: Trade data with columns:
            - timestamp, price, quantity, is_buyer_maker
        d_price: Fractional diff order for price (auto-detect if None)
        d_volume: Fractional diff order for volume (auto-detect if None)
        fdi_window: Window for FDI calculation
    
    Returns:
        DataFrame with engineered features
    """
    result = lob_df.copy()
    
    # Fractional differentiation of price
    if "mid_price" in result.columns:
        log_price = np.log(result["mid_price"])
        
        if d_price is None:
            frac_result = find_optimal_d(log_price)
            d_price = frac_result.d
            logger.info(f"Optimal d for price: {d_price}")
        
        result["price_ffd"] = frac_diff_ffd(log_price, d_price)
    
    # Volume differentiation (if available)
    if "volume" in result.columns:
        log_volume = np.log(result["volume"] + 1)
        
        if d_volume is None:
            frac_result = find_optimal_d(log_volume)
            d_volume = frac_result.d
            logger.info(f"Optimal d for volume: {d_volume}")
        
        result["volume_ffd"] = frac_diff_ffd(log_volume, d_volume)
    
    # OFI features
    if all(f"bid_qty_{i}" in result.columns for i in range(5)):
        result = _add_ofi_features(result)
    
    # FDI
    if "mid_price" in result.columns:
        result["fdi"] = calculate_fdi_series(result["mid_price"], window=fdi_window)
    
    # Volatility
    if "mid_price" in result.columns:
        returns = np.log(result["mid_price"]).diff()
        result["volatility"] = returns.rolling(100).std()
    
    # Trade imbalance (if trade data available)
    if trades_df is not None:
        result = _add_trade_imbalance(result, trades_df)
    
    return result.dropna()


def _add_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OFI features to DataFrame."""
    result = df.copy()
    
    # Calculate bid/ask depth at different levels
    for level in [1, 5, 10, 20]:
        bid_cols = [f"bid_qty_{i}" for i in range(level) if f"bid_qty_{i}" in df.columns]
        ask_cols = [f"ask_qty_{i}" for i in range(level) if f"ask_qty_{i}" in df.columns]
        
        if bid_cols and ask_cols:
            bid_depth = df[bid_cols].sum(axis=1)
            ask_depth = df[ask_cols].sum(axis=1)
            
            result[f"ofi_level_{level}"] = bid_depth - ask_depth
            result[f"ofi_diff_{level}"] = result[f"ofi_level_{level}"].diff()
    
    # Rolling OFI
    for window in ["100ms", "1s", "5s"]:
        if "ofi_diff_1" in result.columns:
            result[f"ofi_{window}"] = result["ofi_diff_1"].rolling(window).sum()
            
            # Z-score
            rolling_mean = result[f"ofi_{window}"].rolling("30s").mean()
            rolling_std = result[f"ofi_{window}"].rolling("30s").std().replace(0, 1e-8)
            result[f"ofi_zscore_{window}"] = (
                result[f"ofi_{window}"] - rolling_mean
            ) / rolling_std
    
    return result


def _add_trade_imbalance(lob_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Add trade imbalance features from trade data."""
    # Calculate signed volume for each trade
    trades = trades_df.copy()
    trades["signed_volume"] = trades["quantity"].where(
        ~trades["is_buyer_maker"],
        -trades["quantity"]
    )
    
    # Resample to match LOB timestamps
    # This is a simplified version; production would need proper alignment
    trades["timestamp_dt"] = pd.to_datetime(trades["timestamp"], unit="ms")
    trades = trades.set_index("timestamp_dt")
    
    trade_imbalance = trades["signed_volume"].rolling("1s").sum()
    
    # Merge with LOB data
    lob_df = lob_df.copy()
    if "timestamp" in lob_df.columns:
        lob_df["timestamp_dt"] = pd.to_datetime(lob_df["timestamp"], unit="ms")
        lob_df = lob_df.set_index("timestamp_dt")
    
    lob_df["trade_imbalance"] = trade_imbalance.reindex(lob_df.index, method="ffill")
    
    return lob_df.reset_index()
