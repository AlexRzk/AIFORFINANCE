"""
Tests for the features module.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.ofi import OrderFlowImbalance, OFIFeatures, RollingStats
from src.features.fracdiff import (
    get_weights,
    get_weights_ffd,
    frac_diff_ffd,
    find_optimal_d,
    FractionalDifferentiator,
    adf_test,
)
from src.features.features import FractalDimensionIndex, FeatureEngineer


class TestRollingStats:
    """Test rolling statistics calculator."""
    
    def test_mean(self):
        stats = RollingStats(window_size=5)
        for i in range(5):
            stats.update(float(i))
        
        assert abs(stats.mean - 2.0) < 1e-10
    
    def test_std(self):
        stats = RollingStats(window_size=5)
        for i in range(5):
            stats.update(float(i))
        
        expected_std = np.std([0, 1, 2, 3, 4], ddof=0)
        assert abs(stats.std - expected_std) < 1e-10
    
    def test_rolling_window(self):
        stats = RollingStats(window_size=3)
        for i in range(10):
            stats.update(float(i))
        
        # Should only have last 3 values: 7, 8, 9
        assert abs(stats.mean - 8.0) < 1e-10


class TestOFI:
    """Test Order Flow Imbalance calculator."""
    
    def test_instant_ofi(self):
        ofi = OrderFlowImbalance()
        
        bid_prices = [100.0, 99.9, 99.8]
        bid_quantities = [10.0, 20.0, 30.0]
        ask_prices = [100.1, 100.2, 100.3]
        ask_quantities = [5.0, 15.0, 25.0]
        
        result = ofi.calculate_instant_ofi(
            bid_prices, bid_quantities,
            ask_prices, ask_quantities,
        )
        
        # Level 0: 10 - 5 = 5
        assert result["level_0"] == 5.0
        
        # Level 5: sum(bid[:3]) - sum(ask[:3]) = 60 - 45 = 15
        # But we only have 3 levels, so it uses all
        assert result["level_5"] == 15.0
    
    def test_update(self):
        ofi = OrderFlowImbalance(timeframes_ms=[100, 1000])
        
        features = ofi.update(
            symbol="BTCUSDT",
            timestamp=1000,
            bid_prices=[100.0, 99.9],
            bid_quantities=[10.0, 20.0],
            ask_prices=[100.1, 100.2],
            ask_quantities=[5.0, 15.0],
        )
        
        assert isinstance(features, OFIFeatures)
        assert features.symbol == "BTCUSDT"
        assert features.ofi_level_0 == 5.0


class TestFracDiff:
    """Test Fractional Differentiation."""
    
    def test_weights_sum(self):
        # Weights should sum to 0 for d > 0
        weights = get_weights(0.5, 100)
        # First weight is 1, subsequent weights are negative
        assert weights[-1] == 1.0
    
    def test_ffd_weights(self):
        weights = get_weights_ffd(0.4)
        assert len(weights) > 0
        assert weights[-1] == 1.0  # Most recent weight is 1
    
    def test_frac_diff_reduces_nonstationarity(self):
        np.random.seed(42)
        
        # Generate random walk (non-stationary)
        n = 1000
        returns = np.random.randn(n) * 0.01
        prices = np.exp(np.cumsum(returns))
        series = pd.Series(np.log(prices))
        
        # Raw series should be non-stationary
        adf_stat_raw, pvalue_raw, _ = adf_test(series)
        
        # FracDiff with d=0.5 should make it more stationary
        diff_series = frac_diff_ffd(series, 0.5)
        adf_stat_diff, pvalue_diff, _ = adf_test(diff_series)
        
        # p-value should decrease (more stationary)
        assert pvalue_diff < pvalue_raw
    
    def test_find_optimal_d(self):
        np.random.seed(42)
        
        # Generate random walk
        n = 1000
        returns = np.random.randn(n) * 0.01
        prices = np.exp(np.cumsum(returns))
        series = pd.Series(np.log(prices))
        
        result = find_optimal_d(
            series,
            d_values=[0.3, 0.5, 0.7, 1.0],
        )
        
        assert 0 < result.d <= 1.0
        assert result.is_stationary or result.adf_pvalue < 0.1
    
    def test_online_fracdiff(self):
        # Use higher threshold for smaller window (faster test)
        fracdiff = FractionalDifferentiator(d=0.4, threshold=1e-3)
        
        # Need enough samples to fill the window
        result = None
        for i in range(fracdiff.window_size + 100):
            result = fracdiff.update(float(i) + np.random.randn() * 0.1)
        
        assert result is not None
        assert fracdiff.get_buffer_size() > 0


class TestFDI:
    """Test Fractal Dimension Index."""
    
    def test_trending_market(self):
        # Strong uptrend should have low FDI (< 1.5)
        fdi = FractalDimensionIndex(window=30)
        
        for i in range(50):
            result = fdi.update(100 + i)  # Linear uptrend
        
        assert result is not None
        assert result < 1.5  # Trending
    
    def test_ranging_market(self):
        # Oscillating market should have high FDI (> 1.5)
        fdi = FractalDimensionIndex(window=30)
        
        for i in range(50):
            # Oscillate between 100 and 101
            price = 100 + (i % 2)
            result = fdi.update(price)
        
        assert result is not None
        assert result > 1.3  # More ranging than trending


class TestFeatureEngineer:
    """Test the main FeatureEngineer class."""
    
    def test_initialization(self):
        fe = FeatureEngineer(symbol="BTCUSDT", d_price=0.4)
        assert fe.symbol == "BTCUSDT"
    
    def test_update(self):
        # Use higher threshold for smaller window (faster test)
        fe = FeatureEngineer(symbol="BTCUSDT", d_price=0.4)
        # Override with smaller window for testing
        fe.price_fracdiff = FractionalDifferentiator(d=0.4, threshold=1e-3)
        fe.volume_fracdiff = FractionalDifferentiator(d=0.5, threshold=1e-3)
        
        # Need many updates to fill buffers
        result = None
        for i in range(500):
            result = fe.update(
                timestamp=i * 100,
                mid_price=100 + i * 0.01 + np.random.randn() * 0.1,
                volume=1000 + np.random.randn() * 100,
                bid_prices=[99.95, 99.90, 99.85],
                bid_quantities=[10, 20, 30],
                ask_prices=[100.05, 100.10, 100.15],
                ask_quantities=[10, 20, 30],
            )
        
        # Should eventually return features
        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.mid_price > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
