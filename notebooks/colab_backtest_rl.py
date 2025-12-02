"""
Google Colab Backtesting Notebook for QR-DQN RL Agent.

This notebook:
1. Loads a trained RL agent
2. Fetches real market data (Binance)
3. Runs backtest with comprehensive metrics
4. Visualizes equity curve, drawdown, and trades
5. Computes Sharpe ratio, Sortino, Calmar, win rate, etc.

Usage:
    !git clone https://github.com/AlexRzk/AIFORFINANCE.git
    %run notebooks/colab_backtest_rl.py
"""

# ============================================================================
# CELL 1: IMPORTS & SETUP
# ============================================================================

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try importing ccxt for real data
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("⚠️  ccxt not available, will use synthetic data")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add to path
if 'AIFORFINANCE' in os.getcwd():
    sys.path.insert(0, os.getcwd())
else:
    sys.path.insert(0, '/content/AIFORFINANCE')

print("✅ Setup complete")

# ============================================================================
# CELL 2: IMPORT RL MODULES
# ============================================================================

import torch
import torch.nn as nn
import gymnasium as gym

# Import from Phase 3
from src.rl.agent import QRDQNAgent, QRDQNNetwork
from src.rl.environment import TradingEnv, EnvConfig
from src.rl.backtester import Backtester, BacktestConfig, run_backtest
from src.rl.data_loader import get_price_data, get_statistics, split_data

print("✅ RL modules imported")

# ============================================================================
# CELL 3: LOAD OR TRAIN AGENT
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_or_train_agent(model_path: str = "best_rl_agent.pt", retrain: bool = False):
    """Load existing agent or train new one."""
    
    agent = QRDQNAgent(
        state_dim=80,
        hidden_dims=[256, 256],
        num_quantiles=32,
        batch_size=128,
        buffer_size=50000,
        min_buffer=1000,
        device=DEVICE,
    )
    
    if not retrain and os.path.exists(model_path):
        logger.info(f"Loading agent from {model_path}")
        agent.load(model_path)
        logger.info(f"✅ Agent loaded, train_steps: {agent.train_steps}")
    else:
        logger.info("No saved agent found, training new agent (quick 5min training)...")
        
        # Quick training on synthetic data
        prices = get_price_data("synthetic", n_synthetic=50000)
        env = TradingEnv(EnvConfig(), prices)
        
        state, _ = env.reset()
        for step in range(5000):  # Quick 5k steps
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store(state, action, reward, next_state, terminated or truncated)
            agent.train_step()
            
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state
            
            if (step + 1) % 1000 == 0:
                print(f"  Trained {step+1}/5000 steps")
        
        agent.save(model_path)
        logger.info(f"✅ Agent trained and saved to {model_path}")
    
    return agent

agent = load_or_train_agent()

# ============================================================================
# CELL 4: LOAD DATA
# ============================================================================

def load_backtest_data(data_source: str = "synthetic", days: int = 365):
    """Load data for backtesting."""
    
    logger.info(f"Loading data from: {data_source}")
    
    if data_source == "synthetic":
        prices = get_price_data("synthetic", n_synthetic=100000)
        logger.info(f"Using synthetic data: {len(prices)} points")
    
    elif data_source == "binance":
        if not HAS_CCXT:
            logger.warning("ccxt not available, using synthetic data")
            prices = get_price_data("synthetic", n_synthetic=100000)
        else:
            prices = get_price_data(
                "binance",
                symbol="BTCUSDT",
                interval="1h",
                days=days,
            )
            if prices is None:
                logger.warning("Failed to fetch Binance data, using synthetic")
                prices = get_price_data("synthetic", n_synthetic=100000)
    
    else:
        prices = get_price_data(data_source)
    
    # Print statistics
    stats = get_statistics(prices)
    print(f"""
╔════════════════════════════════════════╗
║        DATA STATISTICS                 ║
╠════════════════════════════════════════╣
║ Candles:             {stats['n_candles']:>7d}   ║
║ Price Range:    {stats['min_price']:>7.0f} - {stats['max_price']:<7.0f}  ║
║ Total Return:        {stats['total_return']*100:>7.2f}% ║
║ Volatility:          {stats['volatility']*100:>7.2f}% ║
║ Sharpe Ratio:        {stats['sharpe']:>7.2f}   ║
╚════════════════════════════════════════╝
    """)
    
    return prices

# Load data (change data_source to "binance" for real data)
DATA_SOURCE = "synthetic"  # Change to "binance" for real data
prices = load_backtest_data(DATA_SOURCE, days=365)

# ============================================================================
# CELL 5: RUN BACKTEST
# ============================================================================

def run_full_backtest(agent, prices, config=None):
    """Run complete backtest and return results."""
    
    if config is None:
        config = BacktestConfig()
    
    backtester = Backtester(config, DEVICE)
    backtester.reset()
    
    logger.info(f"Running backtest on {len(prices)} candles...")
    
    for i in range(len(prices) - 1):
        current_price = prices[i]
        next_price = prices[i + 1]
        
        # Create simple state (normalized price + position + PnL)
        state = np.zeros(80, dtype=np.float32)
        
        # Normalize price by last 20 candles
        window_start = max(0, i - 20)
        price_window = prices[window_start:i+1]
        if len(price_window) > 1:
            mean_price = np.mean(price_window)
            std_price = np.std(price_window)
            state[0] = (current_price - mean_price) / (std_price + 1e-8)
        
        state[1] = backtester.position
        state[2] = backtester.unrealized_pnl / (backtester.balance + 1e-8)
        state[3] = backtester.balance / 10000
        
        # Get action from agent
        action = agent.select_action(state)
        
        # Execute step
        backtester.step(action, current_price, next_price)
        
        if (i + 1) % 25000 == 0:
            logger.info(f"  Step {i+1:,}/{len(prices)-1:,}")
    
    # Compute metrics
    metrics = backtester.compute_metrics()
    
    return metrics, backtester

logger.info("=" * 60)
logger.info("STARTING BACKTEST")
logger.info("=" * 60)

metrics, backtester = run_full_backtest(agent, prices)

print(metrics)

# ============================================================================
# CELL 6: VISUALIZATION
# ============================================================================

def plot_backtest_results(backtester, metrics):
    """Visualize backtest results."""
    
    equity = backtester.get_equity_curve()
    drawdown = backtester.get_drawdown_curve()
    positions = np.array(backtester.position_history)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Equity curve
    ax = axes[0]
    ax.plot(equity, linewidth=1.5, label='Equity', color='#2E86AB')
    ax.fill_between(range(len(equity)), backtester.config.initial_balance, equity,
                     where=(equity >= backtester.config.initial_balance),
                     color='#2E86AB', alpha=0.2, label='Profit')
    ax.fill_between(range(len(equity)), backtester.config.initial_balance, equity,
                     where=(equity < backtester.config.initial_balance),
                     color='#A23B72', alpha=0.2, label='Loss')
    ax.set_title(f'Equity Curve | Total Return: {metrics.total_return*100:+.2f}% | Sharpe: {metrics.sharpe_ratio:.2f}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Equity ($)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Drawdown
    ax = axes[1]
    ax.fill_between(range(len(drawdown)), drawdown, alpha=0.5, color='#F18F01')
    ax.axhline(y=metrics.max_drawdown, color='red', linestyle='--', linewidth=1.5,
               label=f'Max DD: {metrics.max_drawdown*100:.2f}%')
    ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown %', fontsize=10)
    ax.set_ylim(0, max(drawdown) * 1.1)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Positions
    ax = axes[2]
    colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in positions]
    ax.bar(range(len(positions)), positions, color=colors, width=1, alpha=0.6)
    ax.set_title(f'Position Over Time | Win Rate: {metrics.win_rate*100:.1f}% | Trades: {metrics.num_trades}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Position Size', fontsize=10)
    ax.set_xlabel('Time Steps', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Plot saved to backtest_results.png")
    plt.show()

plot_backtest_results(backtester, metrics)

# ============================================================================
# CELL 7: DETAILED TRADE ANALYSIS
# ============================================================================

def analyze_trades(backtester, metrics):
    """Print detailed trade analysis."""
    
    trades = backtester.trades
    
    if len(trades) == 0:
        logger.info("No trades executed")
        return
    
    print(f"""
╔════════════════════════════════════════════════════╗
║        TRADE ANALYSIS ({len(trades)} trades)          ║
╠════════════════════════════════════════════════════╣
║ Win Rate:              {metrics.win_rate*100:>6.1f}%  ║
║ Avg Trade Return:      {metrics.avg_trade_return*100:>6.2f}% ║
║ Best Trade:            {metrics.best_trade*100:>6.2f}% ║
║ Worst Trade:           {metrics.worst_trade*100:>6.2f}% ║
║ Profit Factor:         {metrics.profit_factor:>6.2f}x ║
╚════════════════════════════════════════════════════╝
    """)
    
    # Top trades
    trades_df = pd.DataFrame(trades)
    trades_df['return'] = trades_df['return'] * 100
    
    print("\n📈 Top 5 Winning Trades:")
    print(trades_df.nlargest(5, 'return')[['direction', 'size', 'return']])
    
    print("\n📉 Top 5 Losing Trades:")
    print(trades_df.nsmallest(5, 'return')[['direction', 'size', 'return']])

analyze_trades(backtester, metrics)

# ============================================================================
# CELL 8: SUMMARY & NEXT STEPS
# ============================================================================

print(f"""
╔════════════════════════════════════════════════════════════╗
║            BACKTEST COMPLETE ✅                           ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Agent Performance:                                        ║
║  • Sharpe Ratio:       {metrics.sharpe_ratio:>8.2f}                    ║
║  • Sortino Ratio:      {metrics.sortino_ratio:>8.2f}                    ║
║  • Calmar Ratio:       {metrics.calmar_ratio:>8.2f}                    ║
║  • Total Return:       {metrics.total_return*100:>8.2f}%                  ║
║  • Max Drawdown:       {metrics.max_drawdown*100:>8.2f}%                  ║
║                                                            ║
║  Next Steps:                                               ║
║  1. Try real data: Change DATA_SOURCE = "binance"         ║
║  2. Optimize hyperparams for higher Sharpe               ║
║  3. Train on longer time horizons                         ║
║  4. Combine with TFT for context-aware trading           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
