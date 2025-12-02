"""
Order Flow Imbalance (OFI) Feature Engineering.

OFI measures the net buying/selling pressure at the best bid/ask levels.
This is the primary "alpha" signal for market microstructure-based trading.

Formula: OFI_t = Σ(Vol_bid,i - Vol_ask,i)

References:
- "High-Frequency Trading and Price Discovery" (Brogaard et al., 2014)
- "The Information Content of Order Flow" (Cont et al., 2014)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class OFIFeatures:
    """Container for OFI-derived features."""
    timestamp: int
    symbol: str
    
    # Raw OFI at different depths
    ofi_level_0: float  # Best bid/ask only
    ofi_level_5: float  # Top 5 levels
    ofi_level_10: float  # Top 10 levels
    ofi_level_20: float  # Top 20 levels
    
    # OFI over different timeframes (rolling sums)
    ofi_100ms: float = 0.0
    ofi_1s: float = 0.0
    ofi_5s: float = 0.0
    ofi_30s: float = 0.0
    
    # Normalized OFI (z-score)
    ofi_zscore_100ms: float = 0.0
    ofi_zscore_1s: float = 0.0
    ofi_zscore_5s: float = 0.0
    
    # Derivative features
    ofi_momentum: float = 0.0  # Change in OFI
    ofi_acceleration: float = 0.0  # Change in OFI momentum
    
    # Volume-weighted OFI
    vw_ofi: float = 0.0
    
    # Trade imbalance (from actual trades)
    trade_imbalance: float = 0.0
    trade_imbalance_1s: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "ofi_level_0": self.ofi_level_0,
            "ofi_level_5": self.ofi_level_5,
            "ofi_level_10": self.ofi_level_10,
            "ofi_level_20": self.ofi_level_20,
            "ofi_100ms": self.ofi_100ms,
            "ofi_1s": self.ofi_1s,
            "ofi_5s": self.ofi_5s,
            "ofi_30s": self.ofi_30s,
            "ofi_zscore_100ms": self.ofi_zscore_100ms,
            "ofi_zscore_1s": self.ofi_zscore_1s,
            "ofi_zscore_5s": self.ofi_zscore_5s,
            "ofi_momentum": self.ofi_momentum,
            "ofi_acceleration": self.ofi_acceleration,
            "vw_ofi": self.vw_ofi,
            "trade_imbalance": self.trade_imbalance,
            "trade_imbalance_1s": self.trade_imbalance_1s,
        }


@dataclass
class RollingStats:
    """Rolling statistics calculator for z-score normalization."""
    window_size: int
    values: Deque[float] = field(default_factory=deque)
    _sum: float = 0.0
    _sum_sq: float = 0.0
    
    def update(self, value: float):
        """Add a new value and update statistics."""
        self.values.append(value)
        self._sum += value
        self._sum_sq += value ** 2
        
        if len(self.values) > self.window_size:
            old_value = self.values.popleft()
            self._sum -= old_value
            self._sum_sq -= old_value ** 2
    
    @property
    def mean(self) -> float:
        """Get rolling mean."""
        n = len(self.values)
        return self._sum / n if n > 0 else 0.0
    
    @property
    def std(self) -> float:
        """Get rolling standard deviation."""
        n = len(self.values)
        if n < 2:
            return 1.0  # Avoid division by zero
        
        variance = (self._sum_sq / n) - (self._sum / n) ** 2
        return max(np.sqrt(max(variance, 0)), 1e-8)
    
    def zscore(self, value: float) -> float:
        """Calculate z-score for a value."""
        return (value - self.mean) / self.std


class OrderFlowImbalance:
    """
    Order Flow Imbalance calculator with multi-timeframe support.
    
    Maintains rolling windows of OFI values and computes various
    derived features useful for ML models.
    """
    
    def __init__(
        self,
        timeframes_ms: List[int] = None,
        zscore_window: int = 1000,
        max_history_ms: int = 60000,  # 1 minute of history
    ):
        """
        Initialize the OFI calculator.
        
        Args:
            timeframes_ms: List of timeframes in milliseconds [100, 1000, 5000]
            zscore_window: Number of samples for z-score normalization
            max_history_ms: Maximum history to maintain in milliseconds
        """
        self.timeframes_ms = timeframes_ms or [100, 1000, 5000, 30000]
        self.zscore_window = zscore_window
        self.max_history_ms = max_history_ms
        
        # History buffers per symbol
        self._ofi_history: Dict[str, Deque[tuple]] = {}  # (timestamp, ofi)
        self._trade_history: Dict[str, Deque[tuple]] = {}  # (timestamp, signed_volume)
        
        # Rolling statistics for z-score
        self._rolling_stats: Dict[str, Dict[int, RollingStats]] = {}
        
        # Previous values for momentum calculation
        self._prev_ofi: Dict[str, float] = {}
        self._prev_momentum: Dict[str, float] = {}
    
    def _init_symbol(self, symbol: str):
        """Initialize buffers for a new symbol."""
        if symbol not in self._ofi_history:
            self._ofi_history[symbol] = deque()
            self._trade_history[symbol] = deque()
            self._rolling_stats[symbol] = {
                tf: RollingStats(window_size=self.zscore_window)
                for tf in self.timeframes_ms
            }
            self._prev_ofi[symbol] = 0.0
            self._prev_momentum[symbol] = 0.0
    
    def _cleanup_history(self, symbol: str, current_ts: int):
        """Remove old entries from history buffers."""
        cutoff = current_ts - self.max_history_ms
        
        while self._ofi_history[symbol] and self._ofi_history[symbol][0][0] < cutoff:
            self._ofi_history[symbol].popleft()
        
        while self._trade_history[symbol] and self._trade_history[symbol][0][0] < cutoff:
            self._trade_history[symbol].popleft()
    
    def calculate_instant_ofi(
        self,
        bid_prices: List[float],
        bid_quantities: List[float],
        ask_prices: List[float],
        ask_quantities: List[float],
        prev_bid_prices: Optional[List[float]] = None,
        prev_bid_quantities: Optional[List[float]] = None,
        prev_ask_prices: Optional[List[float]] = None,
        prev_ask_quantities: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate instantaneous OFI at different depth levels.
        
        This implements the "differential OFI" which measures changes
        in order book depth rather than absolute levels.
        
        Returns:
            Dictionary with OFI at levels 0, 5, 10, 20
        """
        results = {}
        
        for depth, suffix in [(1, "level_0"), (5, "level_5"), (10, "level_10"), (20, "level_20")]:
            # Current depth
            bid_depth = sum(bid_quantities[:depth]) if len(bid_quantities) >= depth else sum(bid_quantities)
            ask_depth = sum(ask_quantities[:depth]) if len(ask_quantities) >= depth else sum(ask_quantities)
            
            if prev_bid_quantities is not None and prev_ask_quantities is not None:
                # Differential OFI: change in depth
                prev_bid_depth = sum(prev_bid_quantities[:depth]) if len(prev_bid_quantities) >= depth else sum(prev_bid_quantities)
                prev_ask_depth = sum(prev_ask_quantities[:depth]) if len(prev_ask_quantities) >= depth else sum(prev_ask_quantities)
                
                # OFI = ΔBid - ΔAsk (positive = buying pressure)
                ofi = (bid_depth - prev_bid_depth) - (ask_depth - prev_ask_depth)
            else:
                # Absolute OFI: raw imbalance
                ofi = bid_depth - ask_depth
            
            results[suffix] = ofi
        
        return results
    
    def calculate_volume_weighted_ofi(
        self,
        bid_prices: List[float],
        bid_quantities: List[float],
        ask_prices: List[float],
        ask_quantities: List[float],
    ) -> float:
        """
        Calculate volume-weighted OFI.
        
        Weights each level by its distance from mid-price,
        giving more importance to levels close to the spread.
        """
        if not bid_prices or not ask_prices:
            return 0.0
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        weighted_bid = 0.0
        weighted_ask = 0.0
        
        for price, qty in zip(bid_prices, bid_quantities):
            distance = abs(mid_price - price)
            weight = 1.0 / (1.0 + distance / mid_price * 100)  # Decay with distance
            weighted_bid += qty * weight
        
        for price, qty in zip(ask_prices, ask_quantities):
            distance = abs(price - mid_price)
            weight = 1.0 / (1.0 + distance / mid_price * 100)
            weighted_ask += qty * weight
        
        return weighted_bid - weighted_ask
    
    def update(
        self,
        symbol: str,
        timestamp: int,
        bid_prices: List[float],
        bid_quantities: List[float],
        ask_prices: List[float],
        ask_quantities: List[float],
        prev_bid_prices: Optional[List[float]] = None,
        prev_bid_quantities: Optional[List[float]] = None,
        prev_ask_prices: Optional[List[float]] = None,
        prev_ask_quantities: Optional[List[float]] = None,
    ) -> OFIFeatures:
        """
        Update OFI calculations with a new order book snapshot.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Unix timestamp in milliseconds
            bid_prices: List of bid prices (descending)
            bid_quantities: List of bid quantities
            ask_prices: List of ask prices (ascending)
            ask_quantities: List of ask quantities
            prev_*: Previous order book state for differential OFI
        
        Returns:
            OFIFeatures with all calculated features
        """
        self._init_symbol(symbol)
        self._cleanup_history(symbol, timestamp)
        
        # Calculate instant OFI
        instant_ofi = self.calculate_instant_ofi(
            bid_prices, bid_quantities,
            ask_prices, ask_quantities,
            prev_bid_prices, prev_bid_quantities,
            prev_ask_prices, prev_ask_quantities,
        )
        
        # Store in history
        self._ofi_history[symbol].append((timestamp, instant_ofi["level_0"]))
        
        # Calculate rolling OFI for each timeframe
        rolling_ofi = {}
        for tf in self.timeframes_ms:
            cutoff = timestamp - tf
            ofi_sum = sum(
                ofi for ts, ofi in self._ofi_history[symbol]
                if ts >= cutoff
            )
            rolling_ofi[tf] = ofi_sum
            
            # Update rolling stats for z-score
            self._rolling_stats[symbol][tf].update(ofi_sum)
        
        # Calculate z-scores
        zscores = {}
        for tf in self.timeframes_ms:
            zscores[tf] = self._rolling_stats[symbol][tf].zscore(rolling_ofi[tf])
        
        # Calculate momentum and acceleration
        current_ofi = instant_ofi["level_0"]
        momentum = current_ofi - self._prev_ofi[symbol]
        acceleration = momentum - self._prev_momentum[symbol]
        
        self._prev_ofi[symbol] = current_ofi
        self._prev_momentum[symbol] = momentum
        
        # Volume-weighted OFI
        vw_ofi = self.calculate_volume_weighted_ofi(
            bid_prices, bid_quantities,
            ask_prices, ask_quantities,
        )
        
        # Build features
        features = OFIFeatures(
            timestamp=timestamp,
            symbol=symbol,
            ofi_level_0=instant_ofi["level_0"],
            ofi_level_5=instant_ofi["level_5"],
            ofi_level_10=instant_ofi["level_10"],
            ofi_level_20=instant_ofi["level_20"],
            ofi_100ms=rolling_ofi.get(100, 0.0),
            ofi_1s=rolling_ofi.get(1000, 0.0),
            ofi_5s=rolling_ofi.get(5000, 0.0),
            ofi_30s=rolling_ofi.get(30000, 0.0),
            ofi_zscore_100ms=zscores.get(100, 0.0),
            ofi_zscore_1s=zscores.get(1000, 0.0),
            ofi_zscore_5s=zscores.get(5000, 0.0),
            ofi_momentum=momentum,
            ofi_acceleration=acceleration,
            vw_ofi=vw_ofi,
        )
        
        return features
    
    def add_trade(
        self,
        symbol: str,
        timestamp: int,
        quantity: float,
        is_buyer_maker: bool,
    ):
        """
        Add a trade to update trade imbalance features.
        
        Args:
            symbol: Trading pair
            timestamp: Trade timestamp in ms
            quantity: Trade quantity
            is_buyer_maker: True if sell aggressor (taker sold)
        """
        self._init_symbol(symbol)
        
        # Signed volume: positive for buys, negative for sells
        signed_volume = quantity if not is_buyer_maker else -quantity
        self._trade_history[symbol].append((timestamp, signed_volume))
    
    def get_trade_imbalance(self, symbol: str, current_ts: int, window_ms: int = 1000) -> float:
        """
        Calculate trade imbalance over a time window.
        
        Returns the net signed volume (buys - sells).
        """
        if symbol not in self._trade_history:
            return 0.0
        
        cutoff = current_ts - window_ms
        imbalance = sum(
            vol for ts, vol in self._trade_history[symbol]
            if ts >= cutoff
        )
        return imbalance


def calculate_ofi_from_dataframe(
    df: pd.DataFrame,
    depth_levels: List[int] = [1, 5, 10, 20],
    timeframes: List[str] = ["100ms", "1s", "5s"],
) -> pd.DataFrame:
    """
    Calculate OFI features from a DataFrame of order book snapshots.
    
    Args:
        df: DataFrame with columns:
            - timestamp: Unix timestamp in ms
            - bid_price_0, bid_qty_0, ..., bid_price_19, bid_qty_19
            - ask_price_0, ask_qty_0, ..., ask_price_19, ask_qty_19
        depth_levels: List of depth levels to calculate OFI for
        timeframes: List of timeframes for rolling OFI
    
    Returns:
        DataFrame with OFI features added
    """
    result = df.copy()
    
    # Calculate bid and ask depths at each level
    for level in depth_levels:
        bid_cols = [f"bid_qty_{i}" for i in range(level)]
        ask_cols = [f"ask_qty_{i}" for i in range(level)]
        
        bid_depth = result[bid_cols].sum(axis=1)
        ask_depth = result[ask_cols].sum(axis=1)
        
        # Instant OFI (absolute)
        result[f"ofi_level_{level}"] = bid_depth - ask_depth
        
        # Differential OFI
        result[f"ofi_diff_level_{level}"] = (
            bid_depth.diff() - ask_depth.diff()
        ).fillna(0)
    
    # Rolling OFI over different timeframes
    # Assuming the DataFrame is indexed by timestamp
    if "timestamp" in result.columns:
        result = result.set_index(pd.to_datetime(result["timestamp"], unit="ms"))
    
    for tf in timeframes:
        for level in depth_levels:
            col = f"ofi_diff_level_{level}"
            result[f"ofi_{tf}_level_{level}"] = result[col].rolling(tf).sum()
    
    # Z-scores
    for tf in timeframes:
        for level in depth_levels:
            col = f"ofi_{tf}_level_{level}"
            rolling_mean = result[col].rolling("30s").mean()
            rolling_std = result[col].rolling("30s").std().replace(0, 1e-8)
            result[f"ofi_zscore_{tf}_level_{level}"] = (
                (result[col] - rolling_mean) / rolling_std
            )
    
    # Reset index
    result = result.reset_index(drop=True)
    
    return result
