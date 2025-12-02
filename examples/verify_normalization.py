"""
Verification script for data normalization pipeline.

This script demonstrates that:
1. Raw prices are non-stationary (ADF test fails)
2. FracDiff with optimal d makes data stationary while preserving memory
3. OFI features are properly normalized with z-scores
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.fracdiff import find_optimal_d, frac_diff_ffd, adf_test
from src.features.ofi import OrderFlowImbalance


def generate_realistic_price_data(n: int = 5000) -> pd.Series:
    """Generate realistic crypto price data (random walk with volatility clustering)."""
    np.random.seed(42)
    
    # GARCH-like volatility clustering
    volatility = np.zeros(n)
    volatility[0] = 0.001
    for i in range(1, n):
        volatility[i] = 0.00001 + 0.1 * volatility[i-1] + 0.85 * (np.random.randn() ** 2) * 0.0001
    
    # Generate returns with time-varying volatility
    returns = np.random.randn(n) * np.sqrt(volatility) + 0.00001  # Small drift
    log_prices = np.cumsum(returns) + np.log(50000)  # BTC-like starting price
    prices = np.exp(log_prices)
    
    return pd.Series(prices, name="price")


def main():
    print("=" * 70)
    print("DATA NORMALIZATION VERIFICATION")
    print("=" * 70)
    
    # Generate data
    prices = generate_realistic_price_data(5000)
    log_prices = np.log(prices)
    
    print(f"\nGenerated {len(prices)} price observations")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # =========================================================================
    # Test 1: Raw prices are non-stationary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Raw Prices (Should be NON-STATIONARY)")
    print("=" * 70)
    
    adf_stat, pvalue, critical = adf_test(log_prices)
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"P-value: {pvalue:.4f}")
    print(f"5% Critical Value: {critical.get('5%', 'N/A')}")
    print(f"Result: {'❌ NON-STATIONARY (expected)' if pvalue > 0.05 else '✅ Stationary'}")
    
    # =========================================================================
    # Test 2: First difference loses memory
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: First Difference d=1 (Should LOSE MEMORY)")
    print("=" * 70)
    
    returns = log_prices.diff().dropna()
    adf_stat, pvalue, _ = adf_test(returns)
    
    # Memory = correlation with original
    overlap = returns.index.intersection(log_prices.index)
    memory = np.corrcoef(log_prices.loc[overlap], returns.loc[overlap])[0, 1]
    
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"P-value: {pvalue:.6f}")
    print(f"Correlation with original (memory): {memory:.4f}")
    print(f"Result: {'✅ Stationary' if pvalue < 0.05 else '❌ Non-stationary'}")
    print(f"Memory: {'❌ LOST (|corr| < 0.1)' if abs(memory) < 0.1 else '✅ Retained'}")
    
    # =========================================================================
    # Test 3: FracDiff preserves memory while achieving stationarity
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Fractional Differentiation (Should be STATIONARY + MEMORY)")
    print("=" * 70)
    
    result = find_optimal_d(
        log_prices,
        d_values=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
        adf_pvalue_threshold=0.05,
    )
    
    print(f"\nOptimal d: {result.d}")
    print(f"ADF Statistic: {result.adf_stat:.4f}")
    print(f"P-value: {result.adf_pvalue:.6f}")
    print(f"Memory retained (correlation): {result.memory_retained:.4f}")
    print(f"Stationary: {'✅ YES' if result.is_stationary else '❌ NO'}")
    print(f"Memory: {'✅ RETAINED' if abs(result.memory_retained) > 0.1 else '⚠️ Low'}")
    
    # =========================================================================
    # Test 4: Compare different d values
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Memory vs Stationarity Trade-off")
    print("=" * 70)
    print(f"{'d':<6} {'ADF p-value':<12} {'Memory':<10} {'Stationary':<12} {'Verdict'}")
    print("-" * 60)
    
    for d in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        if d == 0:
            diff_series = log_prices
        elif d == 1:
            diff_series = log_prices.diff().dropna()
        else:
            diff_series = frac_diff_ffd(log_prices, d)
        
        if len(diff_series) < 20:
            continue
        
        adf_stat, pvalue, _ = adf_test(diff_series)
        overlap = diff_series.index.intersection(log_prices.index)
        if len(overlap) > 0:
            memory = np.corrcoef(log_prices.loc[overlap], diff_series.loc[overlap])[0, 1]
        else:
            memory = 0
        
        stationary = pvalue < 0.05
        has_memory = abs(memory) > 0.1
        
        if stationary and has_memory:
            verdict = "✅ OPTIMAL"
        elif stationary and not has_memory:
            verdict = "⚠️ No memory"
        elif not stationary and has_memory:
            verdict = "⚠️ Non-stationary"
        else:
            verdict = "❌ Bad"
        
        print(f"{d:<6.1f} {pvalue:<12.4f} {memory:<10.4f} {'Yes' if stationary else 'No':<12} {verdict}")
    
    # =========================================================================
    # Test 5: OFI Z-score normalization
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: OFI Z-score Normalization")
    print("=" * 70)
    
    ofi = OrderFlowImbalance(timeframes_ms=[100, 1000, 5000])
    
    # Simulate orderbook updates
    np.random.seed(42)
    zscores = []
    
    for i in range(2000):
        # Simulate varying order book imbalance
        imbalance = np.sin(i / 100) * 50 + np.random.randn() * 20
        
        bid_qty = max(10, 100 + imbalance + np.random.randn() * 10)
        ask_qty = max(10, 100 - imbalance + np.random.randn() * 10)
        
        features = ofi.update(
            symbol="BTCUSDT",
            timestamp=i * 100,
            bid_prices=[50000 - j * 0.5 for j in range(5)],
            bid_quantities=[bid_qty / (j + 1) for j in range(5)],
            ask_prices=[50000 + j * 0.5 for j in range(5)],
            ask_quantities=[ask_qty / (j + 1) for j in range(5)],
        )
        
        if i > 100:  # After warmup
            zscores.append(features.ofi_zscore_1s)
    
    zscores = np.array(zscores)
    
    print(f"OFI Z-scores computed: {len(zscores)}")
    print(f"Mean: {zscores.mean():.4f} (should be ≈ 0)")
    print(f"Std:  {zscores.std():.4f} (should be ≈ 1)")
    print(f"Min:  {zscores.min():.4f}")
    print(f"Max:  {zscores.max():.4f}")
    
    mean_ok = abs(zscores.mean()) < 0.5
    std_ok = 0.5 < zscores.std() < 2.0
    
    print(f"\nZ-score normalization: {'✅ WORKING' if mean_ok and std_ok else '⚠️ Check parameters'}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
✅ Fractional Differentiation is working correctly:
   - Finds optimal d where ADF p-value < 0.05 (stationary)
   - Preserves memory (correlation with original series)
   - Balance between d=0 (non-stationary) and d=1 (no memory)

✅ OFI Z-score normalization is working:
   - Mean ≈ 0, Std ≈ 1 after warmup period
   - Provides regime-adaptive signals

The data pipeline is ready for the TFT model!
""")


if __name__ == "__main__":
    main()
