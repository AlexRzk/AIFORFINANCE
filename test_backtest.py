#!/usr/bin/env python
"""Quick test of backtester"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.rl.data_loader import get_price_data, get_statistics, split_data
from src.rl.backtester import Backtester, BacktestConfig

print("=" * 60)
print("BACKTESTER TEST")
print("=" * 60)

# Test 1: Data loading
print("\n1️⃣  Testing data loading...")
prices = get_price_data('synthetic', n_synthetic=10000)
stats = get_statistics(prices)
print(f"   ✅ Loaded {len(prices)} prices")
print(f"   Sharpe: {stats['sharpe']:.2f}")

# Test 2: Backtester
print("\n2️⃣  Testing backtester...")
config = BacktestConfig(initial_balance=10000.0)
backtester = Backtester(config, device='cpu')

for i in range(len(prices) - 1):
    action = np.random.randint(0, 3)  # Random action
    backtester.step(action, prices[i], prices[i+1])

metrics = backtester.compute_metrics()
print(f"   ✅ Backtest completed")
print(f"   Total return: {metrics.total_return*100:+.2f}%")
print(f"   Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"   Max DD: {metrics.max_drawdown_pct*100:.2f}%")
print(f"   Trades: {metrics.num_trades}")

# Test 3: Agent integration (simple mock)
print("\n3️⃣  Testing agent strategy with backtest...")

backtester.reset()

for i in range(min(1000, len(prices) - 1)):
    # Simple momentum strategy (mock agent)
    if i < 2:
        action = 0
    else:
        price_change = (prices[i] - prices[i-1]) / prices[i-1]
        if price_change > 0.001:
            action = 1  # Buy on uptrend
        elif price_change < -0.001:
            action = 2  # Sell on downtrend
        else:
            action = 0  # Hold
    
    backtester.step(action, prices[i], prices[i+1])

metrics = backtester.compute_metrics()
print(f"   ✅ Strategy backtest completed")
print(f"   Total return: {metrics.total_return*100:+.2f}%")
print(f"   Trades: {metrics.num_trades}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
