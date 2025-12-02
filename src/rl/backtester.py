"""
Backtesting framework for RL trading agent with comprehensive metrics.

Metrics computed:
- Sharpe Ratio (annualized)
- Sortino Ratio (downside deviation)
- Calmar Ratio (return / max drawdown)
- Maximum Drawdown
- Win Rate
- Profit Factor
- Total Return
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_balance: float = 10000.0
    max_position: float = 1.0
    position_step: float = 0.25
    trading_fee: float = 0.0004
    slippage: float = 0.0001
    max_drawdown: float = 0.2
    risk_free_rate: float = 0.02  # Annual


@dataclass
class BacktestMetrics:
    """Results from backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        return f"""
╔════════════════════════════════════════╗
║        BACKTEST RESULTS                ║
╠════════════════════════════════════════╣
║ Total Return:        {self.total_return*100:>7.2f}% ║
║ Sharpe Ratio:        {self.sharpe_ratio:>7.2f}   ║
║ Sortino Ratio:       {self.sortino_ratio:>7.2f}   ║
║ Calmar Ratio:        {self.calmar_ratio:>7.2f}   ║
║ Max Drawdown:        {self.max_drawdown_pct*100:>7.2f}% ║
║ Win Rate:            {self.win_rate*100:>7.2f}% ║
║ Profit Factor:       {self.profit_factor:>7.2f}x ║
║ Num Trades:          {self.num_trades:>7d}   ║
║ Avg Trade Return:    {self.avg_trade_return*100:>7.2f}% ║
║ Best Trade:          {self.best_trade*100:>7.2f}% ║
║ Worst Trade:         {self.worst_trade*100:>7.2f}% ║
╚════════════════════════════════════════╝
        """


class Backtester:
    """Backtesting engine for RL trading agent."""
    
    def __init__(self, config: BacktestConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        
        # State
        self.balance = config.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_balance = config.initial_balance
        
        # History
        self.balance_history = [config.initial_balance]
        self.equity_history = [config.initial_balance]
        self.position_history = [0.0]
        self.returns_history = []
        self.trade_returns = []
        self.drawdown_history = []
        self.trades = []  # (entry_price, exit_price, position_size, return)
        
    def reset(self):
        """Reset backtester state."""
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_balance = self.config.initial_balance
        
        self.balance_history = [self.config.initial_balance]
        self.equity_history = [self.config.initial_balance]
        self.position_history = [0.0]
        self.returns_history = []
        self.trade_returns = []
        self.drawdown_history = []
        self.trades = []
    
    def step(self, action: int, current_price: float, next_price: float) -> Dict:
        """Execute one step of backtest.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: Price at decision time
            next_price: Price at next step
            
        Returns:
            Dict with step metrics
        """
        old_position = self.position
        
        # Execute action
        if action == 1:  # Buy
            new_pos = min(self.position + self.config.position_step, self.config.max_position)
        elif action == 2:  # Sell
            new_pos = max(self.position - self.config.position_step, -self.config.max_position)
        else:
            new_pos = self.position
        
        # Trading cost
        if new_pos != self.position:
            cost = abs(new_pos - self.position) * self.balance * (
                self.config.trading_fee + self.config.slippage
            )
            self.balance -= cost
            self.realized_pnl -= cost
        
        # Handle position change
        if self.position != 0 and np.sign(new_pos) != np.sign(self.position):
            # Close existing position
            self.balance += self.unrealized_pnl
            self.realized_pnl += self.unrealized_pnl
            
            # Record trade
            trade_return = self.unrealized_pnl / (abs(self.position) * self.entry_price * self.balance)
            self.trade_returns.append(trade_return)
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'size': abs(self.position),
                'direction': 'long' if self.position > 0 else 'short',
                'return': trade_return
            })
            
            self.unrealized_pnl = 0.0
            self.entry_price = current_price if new_pos != 0 else 0.0
        elif self.position == 0 and new_pos != 0:
            # Open new position
            self.entry_price = current_price
        
        self.position = new_pos
        
        # Update PnL at next price
        if self.position != 0:
            price_change = (next_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance
        else:
            self.unrealized_pnl = 0.0
        
        # Calculate step return
        total_equity = self.balance + self.unrealized_pnl
        if len(self.equity_history) > 0:
            step_return = (total_equity - self.equity_history[-1]) / self.equity_history[-1]
        else:
            step_return = 0.0
        
        self.returns_history.append(step_return)
        
        # Update max balance for drawdown
        self.max_balance = max(self.max_balance, total_equity)
        
        # Calculate drawdown
        drawdown = (self.max_balance - total_equity) / self.max_balance if self.max_balance > 0 else 0
        self.drawdown_history.append(drawdown)
        
        # Record history
        self.balance_history.append(self.balance)
        self.equity_history.append(total_equity)
        self.position_history.append(self.position)
        
        return {
            "balance": self.balance,
            "equity": total_equity,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "step_return": step_return,
            "drawdown": drawdown,
        }
    
    def compute_metrics(self) -> BacktestMetrics:
        """Compute all backtest metrics."""
        if len(self.equity_history) < 2:
            return BacktestMetrics(
                total_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                win_rate=0.0, profit_factor=1.0, num_trades=len(self.trades),
                avg_trade_return=0.0, best_trade=0.0, worst_trade=0.0
            )
        
        # Total return
        final_equity = self.equity_history[-1]
        total_return = (final_equity - self.config.initial_balance) / self.config.initial_balance
        
        # Sharpe Ratio (annualized)
        returns_array = np.array(self.returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return > 0:
            # Assuming hourly data for crypto (24 hours * 252 trading days)
            sharpe = (mean_return - self.config.risk_free_rate / (24 * 252)) / std_return
            sharpe_annualized = sharpe * np.sqrt(24 * 252)
        else:
            sharpe_annualized = 0.0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
        else:
            downside_std = 0.0
        
        if downside_std > 0:
            sortino = (mean_return - self.config.risk_free_rate / (24 * 252)) / downside_std
            sortino_annualized = sortino * np.sqrt(24 * 252)
        else:
            sortino_annualized = 0.0
        
        # Maximum Drawdown
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
        
        # Calmar Ratio
        if max_drawdown > 0:
            calmar = total_return / max_drawdown
        else:
            calmar = 0.0 if total_return <= 0 else float('inf')
        
        # Win Rate
        if len(self.trade_returns) > 0:
            winning_trades = sum(1 for r in self.trade_returns if r > 0)
            win_rate = winning_trades / len(self.trade_returns)
        else:
            win_rate = 0.0
        
        # Profit Factor (sum of wins / sum of losses)
        winning_sum = sum(r for r in self.trade_returns if r > 0)
        losing_sum = abs(sum(r for r in self.trade_returns if r < 0))
        
        if losing_sum > 0:
            profit_factor = winning_sum / losing_sum if winning_sum > 0 else 0.0
        else:
            profit_factor = float('inf') if winning_sum > 0 else 1.0
        
        # Trade stats
        if len(self.trade_returns) > 0:
            avg_trade_return = np.mean(self.trade_returns)
            best_trade = np.max(self.trade_returns)
            worst_trade = np.min(self.trade_returns)
        else:
            avg_trade_return = 0.0
            best_trade = 0.0
            worst_trade = 0.0
        
        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_annualized,
            sortino_ratio=sortino_annualized,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )
    
    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_history)
    
    def get_drawdown_curve(self) -> np.ndarray:
        """Get drawdown curve."""
        return np.array(self.drawdown_history)


def run_backtest(agent, prices: np.ndarray, config: Optional[BacktestConfig] = None,
                 device: str = "cuda") -> Tuple[BacktestMetrics, Dict]:
    """Run full backtest of agent on price data.
    
    Args:
        agent: Trained QRDQNAgent
        prices: Array of prices
        config: BacktestConfig (uses default if None)
        device: Device to use
        
    Returns:
        Tuple of (metrics, dict with curves)
    """
    if config is None:
        config = BacktestConfig()
    
    backtester = Backtester(config, device)
    backtester.reset()
    
    logger.info(f"Starting backtest on {len(prices)} price points...")
    
    for i in range(len(prices) - 1):
        current_price = prices[i]
        next_price = prices[i + 1]
        
        # Get agent action
        # Build state (simplified - just normalized price)
        state = np.zeros(80, dtype=np.float32)
        state[0] = (current_price - prices[max(0, i-20):i].mean()) / (prices[max(0, i-20):i].std() + 1e-8)
        state[1] = backtester.position
        state[2] = backtester.unrealized_pnl / backtester.balance
        
        action = agent.select_action(state)
        
        # Execute step
        backtester.step(action, current_price, next_price)
        
        if (i + 1) % 10000 == 0:
            logger.info(f"  Step {i+1:,}/{len(prices)-1:,}")
    
    # Compute metrics
    metrics = backtester.compute_metrics()
    
    curves = {
        "equity": backtester.get_equity_curve(),
        "drawdown": backtester.get_drawdown_curve(),
        "positions": np.array(backtester.position_history),
        "trades": backtester.trades,
    }
    
    logger.info(metrics)
    
    return metrics, curves
