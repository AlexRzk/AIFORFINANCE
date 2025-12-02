"""
Google Colab Backtesting Notebook for QR-DQN RL Agent.

This notebook is SELF-CONTAINED - no external imports needed.
Uses the same agent architecture as colab_train_rl.py.

Usage:
    !git clone https://github.com/AlexRzk/AIFORFINANCE.git
    %cd AIFORFINANCE
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print("✅ Setup complete")

# ============================================================================
# CELL 2: TORCH IMPORTS
# ============================================================================

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Using device: {DEVICE}")

# ============================================================================
# CELL 3: POPART NORMALIZER (same as training script)
# ============================================================================

class PopArtNormalizer(nn.Module):
    """Adaptive reward normalization."""
    
    def __init__(self, input_dim: int, output_dim: int, beta: float = 0.0001):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("std", torch.ones(output_dim))
        self.register_buffer("count", torch.zeros(1))
    
    def forward(self, x):
        return self.linear(x)
    
    def denormalize(self, x):
        return x * self.std + self.mean
    
    @torch.no_grad()
    def update_stats(self, targets):
        batch_mean = targets.mean()
        batch_std = targets.std() + 1e-4
        
        if self.count == 0:
            self.mean.fill_(batch_mean)
            self.std.fill_(batch_std)
        else:
            self.mean.mul_(1 - self.beta).add_(batch_mean * self.beta)
            self.std.mul_(1 - self.beta).add_(batch_std * self.beta)
        
        self.count.add_(1)

# ============================================================================
# CELL 4: QR-DQN NETWORK (same as training script)
# ============================================================================

class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN - models full return distribution."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int], num_actions: int = 3, 
                 num_quantiles: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        self.popart = PopArtNormalizer(in_dim, num_actions * num_quantiles)
        
        taus = torch.linspace(0, 1, num_quantiles + 1)
        self.register_buffer("tau", (taus[:-1] + taus[1:]) / 2)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        features = self.features(state)
        output = self.popart(features)
        return output.view(-1, self.num_actions, self.num_quantiles)
    
    def get_q_values(self, state):
        quantiles = self.forward(state)
        return quantiles.mean(dim=-1)
    
    def get_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        with torch.no_grad():
            return self.get_q_values(state).argmax(dim=-1).item()

# ============================================================================
# CELL 5: REPLAY BUFFER (same as training script)
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
    
    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = torch.from_numpy(state)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = torch.from_numpy(next_state)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (self.states[idx].to(self.device), self.actions[idx].to(self.device),
                self.rewards[idx].to(self.device), self.next_states[idx].to(self.device),
                self.dones[idx].to(self.device))
    
    def __len__(self):
        return self.size

# ============================================================================
# CELL 6: QR-DQN AGENT (same as training script)
# ============================================================================

class QRDQNAgent:
    def __init__(self, state_dim: int, hidden_dims: List[int], num_quantiles: int = 32,
                 lr: float = 3e-4, gamma: float = 0.99, batch_size: int = 256,
                 buffer_size: int = 100000, min_buffer: int = 1000,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000, device: str = "cuda"):
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.num_quantiles = num_quantiles
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        
        self.online = QRDQNNetwork(state_dim, hidden_dims, 3, num_quantiles).to(self.device)
        self.target = QRDQNNetwork(state_dim, hidden_dims, 3, num_quantiles).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=lr, weight_decay=1e-5)
        self.buffer = ReplayBuffer(buffer_size, state_dim, device)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay
        self.train_steps = 0
        self.target_update_freq = 1000
    
    def select_action(self, state, deterministic: bool = False):
        """Select action. Use deterministic=True for backtesting."""
        eps = 0.0 if deterministic else self.epsilon
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online.get_action(state_t, eps)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.buffer) < self.min_buffer:
            return {"loss": 0.0}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        current_q = self.online(states)
        current_q = current_q.gather(1, actions.unsqueeze(-1).expand(-1, 1, self.num_quantiles)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.online.get_q_values(next_states).argmax(dim=-1, keepdim=True)
            next_q = self.target(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(-1).expand(-1, 1, self.num_quantiles)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        td_error = target_q.unsqueeze(1) - current_q.unsqueeze(2)
        huber = torch.where(td_error.abs() <= 1, 0.5 * td_error.pow(2), td_error.abs() - 0.5)
        tau = self.online.tau.view(1, -1, 1)
        loss = (torch.abs(tau - (td_error.detach() < 0).float()) * huber).sum(dim=-1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        
        self.train_steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        if self.train_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        
        return {"loss": loss.item()}
    
    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "train_steps": self.train_steps,
                    "epsilon": self.epsilon}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_steps = ckpt["train_steps"]
        self.epsilon = ckpt["epsilon"]
        logger.info(f"✅ Loaded agent with {self.train_steps} train steps")

# ============================================================================
# CELL 7: BACKTEST METRICS
# ============================================================================

@dataclass
class BacktestMetrics:
    """Results from backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    
    def __str__(self) -> str:
        return f"""
╔════════════════════════════════════════╗
║        BACKTEST RESULTS                ║
╠════════════════════════════════════════╣
║ Total Return:       {self.total_return*100:>8.2f}%           ║
║ Sharpe Ratio:       {self.sharpe_ratio:>8.2f}            ║
║ Sortino Ratio:      {self.sortino_ratio:>8.2f}            ║
║ Calmar Ratio:       {self.calmar_ratio:>8.2f}            ║
║ Max Drawdown:       {self.max_drawdown*100:>8.2f}%           ║
║ Win Rate:           {self.win_rate*100:>8.2f}%           ║
║ Profit Factor:      {self.profit_factor:>8.2f}x           ║
║ Num Trades:         {self.num_trades:>8d}            ║
║ Avg Trade Return:   {self.avg_trade_return*100:>8.4f}%         ║
║ Best Trade:         {self.best_trade*100:>8.2f}%           ║
║ Worst Trade:        {self.worst_trade*100:>8.2f}%           ║
╚════════════════════════════════════════╝
        """

# ============================================================================
# CELL 8: BACKTESTER CLASS
# ============================================================================

@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    max_position: float = 1.0
    position_step: float = 0.25
    trading_fee: float = 0.0004
    slippage: float = 0.0001
    risk_free_rate: float = 0.02


class Backtester:
    """Backtesting engine for RL trading agent."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.max_balance = self.config.initial_balance
        
        self.equity_history = [self.config.initial_balance]
        self.returns_history = []
        self.trade_returns = []
        self.drawdown_history = []
        self.position_history = [0.0]
        self.trades = []
    
    def step(self, action: int, current_price: float, next_price: float) -> Dict:
        old_position = self.position
        
        if action == 1:
            new_pos = min(self.position + self.config.position_step, self.config.max_position)
        elif action == 2:
            new_pos = max(self.position - self.config.position_step, -self.config.max_position)
        else:
            new_pos = self.position
        
        if new_pos != self.position:
            cost = abs(new_pos - self.position) * self.balance * (self.config.trading_fee + self.config.slippage)
            self.balance -= cost
        
        if self.position != 0 and np.sign(new_pos) != np.sign(self.position):
            self.balance += self.unrealized_pnl
            if abs(self.position) > 0.01:
                trade_return = self.unrealized_pnl / (abs(self.position) * self.entry_price * 0.01 + 1e-8)
                self.trade_returns.append(trade_return)
                self.trades.append({'return': trade_return, 'size': abs(self.position)})
            self.unrealized_pnl = 0.0
            self.entry_price = current_price if new_pos != 0 else 0.0
        elif self.position == 0 and new_pos != 0:
            self.entry_price = current_price
        
        self.position = new_pos
        
        if self.position != 0:
            price_change = (next_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance
        else:
            self.unrealized_pnl = 0.0
        
        total_equity = self.balance + self.unrealized_pnl
        
        if len(self.equity_history) > 0:
            step_return = (total_equity - self.equity_history[-1]) / self.equity_history[-1]
        else:
            step_return = 0.0
        
        self.returns_history.append(step_return)
        self.max_balance = max(self.max_balance, total_equity)
        
        drawdown = (self.max_balance - total_equity) / self.max_balance if self.max_balance > 0 else 0
        self.drawdown_history.append(drawdown)
        self.equity_history.append(total_equity)
        self.position_history.append(self.position)
        
        return {"equity": total_equity, "drawdown": drawdown}
    
    def compute_metrics(self) -> BacktestMetrics:
        if len(self.equity_history) < 2:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
        
        final_equity = self.equity_history[-1]
        total_return = (final_equity - self.config.initial_balance) / self.config.initial_balance
        
        returns_array = np.array(self.returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(24 * 252) if std_return > 0 else 0.0
        
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino = (mean_return / (downside_std + 1e-8)) * np.sqrt(24 * 252) if downside_std > 0 else 0.0
        
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        if len(self.trade_returns) > 0:
            win_rate = sum(1 for r in self.trade_returns if r > 0) / len(self.trade_returns)
            winning_sum = sum(r for r in self.trade_returns if r > 0)
            losing_sum = abs(sum(r for r in self.trade_returns if r < 0))
            profit_factor = winning_sum / losing_sum if losing_sum > 0 else 1.0
            avg_trade = np.mean(self.trade_returns)
            best_trade = np.max(self.trade_returns)
            worst_trade = np.min(self.trade_returns)
        else:
            win_rate = 0.0
            profit_factor = 1.0
            avg_trade = 0.0
            best_trade = 0.0
            worst_trade = 0.0
        
        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_return=avg_trade,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

# ============================================================================
# CELL 9: DATA GENERATION
# ============================================================================

def generate_price_data(n_samples: int = 100000) -> np.ndarray:
    """Generate synthetic GARCH-like price data."""
    np.random.seed(42)
    
    volatility = np.zeros(n_samples)
    volatility[0] = 0.001
    for i in range(1, n_samples):
        volatility[i] = 0.00001 + 0.1 * volatility[i-1] + 0.85 * (np.random.randn()**2) * 0.0001
    
    returns = np.random.randn(n_samples) * np.sqrt(volatility) + 0.00001
    prices = np.exp(np.cumsum(returns) + np.log(50000))
    
    return prices.astype(np.float32)


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
    }

# ============================================================================
# CELL 10: LOAD/CREATE AGENT
# ============================================================================

def load_or_train_agent(model_path: str = "best_rl_agent.pt"):
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
    
    if os.path.exists(model_path):
        logger.info(f"Loading agent from {model_path}")
        agent.load(model_path)
    else:
        logger.info("No saved agent found, training quick agent...")
        
        prices = generate_price_data(50000)
        
        # Quick training
        state = np.zeros(80, dtype=np.float32)
        for step in range(10000):
            action = agent.select_action(state)
            next_state = np.zeros(80, dtype=np.float32)
            next_state[0] = np.random.randn() * 0.01
            reward = np.random.randn() * 0.01
            agent.store(state, action, reward, next_state, False)
            agent.train_step()
            state = next_state
            
            if (step + 1) % 2500 == 0:
                print(f"  Quick training: {step+1}/10000 steps")
        
        agent.save(model_path)
        logger.info(f"✅ Agent trained and saved")
    
    return agent


print("\n" + "="*60)
print("LOADING/TRAINING AGENT")
print("="*60)

agent = load_or_train_agent()

# ============================================================================
# CELL 11: RUN BACKTEST
# ============================================================================

print("\n" + "="*60)
print("RUNNING BACKTEST")
print("="*60)

# Generate test data
prices = generate_price_data(100000)
stats = get_statistics(prices)
print(f"Data: {stats['n_candles']} candles, Total return: {stats['total_return']*100:.2f}%")

# Run backtest
config = BacktestConfig()
backtester = Backtester(config)
backtester.reset()

for i in range(len(prices) - 1):
    current_price = prices[i]
    next_price = prices[i + 1]
    
    # Build state
    state = np.zeros(80, dtype=np.float32)
    window_start = max(0, i - 20)
    price_window = prices[window_start:i+1]
    if len(price_window) > 1:
        state[0] = (current_price - np.mean(price_window)) / (np.std(price_window) + 1e-8)
    state[1] = backtester.position
    state[2] = backtester.unrealized_pnl / (backtester.balance + 1e-8)
    
    # Get action (deterministic for backtesting)
    action = agent.select_action(state, deterministic=True)
    
    backtester.step(action, current_price, next_price)
    
    if (i + 1) % 25000 == 0:
        print(f"  Step {i+1:,}/{len(prices)-1:,}")

# Compute and display metrics
metrics = backtester.compute_metrics()
print(metrics)

# ============================================================================
# CELL 12: VISUALIZATION
# ============================================================================

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Equity curve
    ax = axes[0]
    equity = np.array(backtester.equity_history)
    ax.plot(equity, linewidth=1.5, color='#2E86AB')
    ax.axhline(y=config.initial_balance, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(equity)), config.initial_balance, equity,
                     where=(equity >= config.initial_balance), color='green', alpha=0.2)
    ax.fill_between(range(len(equity)), config.initial_balance, equity,
                     where=(equity < config.initial_balance), color='red', alpha=0.2)
    ax.set_title(f'Equity Curve | Return: {metrics.total_return*100:+.2f}% | Sharpe: {metrics.sharpe_ratio:.2f}', fontweight='bold')
    ax.set_ylabel('Equity ($)')
    ax.grid(True, alpha=0.3)
    
    # Drawdown
    ax = axes[1]
    drawdown = np.array(backtester.drawdown_history)
    ax.fill_between(range(len(drawdown)), drawdown * 100, alpha=0.5, color='#F18F01')
    ax.axhline(y=metrics.max_drawdown * 100, color='red', linestyle='--', label=f'Max DD: {metrics.max_drawdown*100:.2f}%')
    ax.set_title('Drawdown Over Time', fontweight='bold')
    ax.set_ylabel('Drawdown %')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Positions
    ax = axes[2]
    positions = np.array(backtester.position_history)
    colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in positions]
    ax.bar(range(len(positions)), positions, color=colors, width=1, alpha=0.6)
    ax.set_title(f'Position | Win Rate: {metrics.win_rate*100:.1f}% | Trades: {metrics.num_trades}', fontweight='bold')
    ax.set_ylabel('Position')
    ax.set_xlabel('Time Step')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Chart saved to backtest_results.png")
    plt.show()
    
except ImportError:
    print("⚠️  matplotlib not available for visualization")

# ============================================================================
# CELL 13: SUMMARY
# ============================================================================

print(f"""
╔════════════════════════════════════════════════════════════╗
║            BACKTEST COMPLETE ✅                            ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Key Metrics:                                              ║
║  • Sharpe Ratio:       {metrics.sharpe_ratio:>8.2f}                         ║
║  • Total Return:       {metrics.total_return*100:>8.2f}%                       ║
║  • Max Drawdown:       {metrics.max_drawdown*100:>8.2f}%                       ║
║  • Win Rate:           {metrics.win_rate*100:>8.2f}%                       ║
║                                                            ║
║  Files:                                                    ║
║  • best_rl_agent.pt - Trained agent                       ║
║  • backtest_results.png - Performance charts              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
