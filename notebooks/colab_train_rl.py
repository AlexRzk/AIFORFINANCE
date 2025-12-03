"""
Google Colab Training Script for QR-DQN RL Agent.

Copy this entire file to a Colab notebook and run!
Supports: T4 (free tier), A100, H100

Usage in Colab:
    1. Copy this file to Colab or clone: !git clone https://github.com/AlexRzk/AIFORFINANCE.git
    2. Run: !python colab_train_rl.py
    
Or run cells individually in a notebook.
"""

# ============================================================================
# CELL 1: SETUP - Run this first!
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages."""
    packages = [
        "torch",
        "numpy",
        "pandas",
        "gymnasium",
        "tensorboard",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ Packages installed!")

# Uncomment to install:
# install_packages()

# ============================================================================
# CELL 2: IMPORTS
# ============================================================================

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CELL 3: GPU DETECTION
# ============================================================================

def get_gpu_info():
    """Get GPU information and optimal settings."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "memory_gb": 0, "settings": {
            "batch_size": 64, "hidden_dims": [128, 128], "num_quantiles": 16, "buffer_size": 10000
        }}
    
    props = torch.cuda.get_device_properties(0)
    name = props.name
    memory_gb = props.total_memory / 1e9
    name_lower = name.lower()
    
    if "t4" in name_lower:
        settings = {"batch_size": 128, "hidden_dims": [256, 256], "num_quantiles": 32, "buffer_size": 50000}
    elif "a100" in name_lower:
        settings = {"batch_size": 512, "hidden_dims": [512, 512, 256], "num_quantiles": 64, "buffer_size": 200000}
    elif "h100" in name_lower:
        settings = {"batch_size": 1024, "hidden_dims": [512, 512, 512], "num_quantiles": 64, "buffer_size": 500000}
    elif "v100" in name_lower:
        settings = {"batch_size": 256, "hidden_dims": [256, 256, 128], "num_quantiles": 32, "buffer_size": 100000}
    else:
        settings = {"batch_size": 256, "hidden_dims": [256, 256, 128], "num_quantiles": 32, "buffer_size": 100000}
    
    return {"name": name, "memory_gb": memory_gb, "settings": settings}

gpu_info = get_gpu_info()
print(f"🖥️  GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
print(f"⚙️  Optimal settings: {gpu_info['settings']}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Using device: {DEVICE}")

# ============================================================================
# CELL 4: TRADING ENVIRONMENT
# ============================================================================

@dataclass
class EnvConfig:
    context_dim: int = 64
    initial_balance: float = 10000.0
    max_position: float = 1.0
    position_step: float = 0.25
    trading_fee: float = 0.0004
    slippage: float = 0.0001
    max_steps: int = 1000
    max_drawdown: float = 0.2
    stop_loss: float = 0.05


class TradingEnv(gym.Env):
    """Crypto Trading Environment with TFT context support."""
    
    def __init__(self, config: EnvConfig, price_data: np.ndarray, tft_model=None, feature_data=None):
        super().__init__()
        self.config = config
        self.price_data = price_data
        self.tft_model = tft_model
        self.feature_data = feature_data
        
        # State: context (64) + account (16) = 80
        self.account_state_dim = 16
        self.state_dim = config.context_dim + self.account_state_dim
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        self._reset_state()
    
    def _reset_state(self):
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.time_in_position = 0
        self.current_step = 0
        self.max_balance = self.balance
        self.recent_returns = np.zeros(10)
        self.recent_volatility = 0.01
        self.returns_history = []
        self.data_start_idx = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        
        max_start = len(self.price_data) - self.config.max_steps - 100
        if max_start > 0:
            self.data_start_idx = self.np_random.integers(0, max_start)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        current_price = self._get_price()
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
            cost = abs(new_pos - self.position) * self.balance * (self.config.trading_fee + self.config.slippage)
            self.balance -= cost
            self.realized_pnl -= cost
        
        # Handle position change
        if self.position != 0 and np.sign(new_pos) != np.sign(self.position):
            self.balance += self.unrealized_pnl
            self.realized_pnl += self.unrealized_pnl
            self.unrealized_pnl = 0.0
            self.entry_price = current_price
        elif self.position == 0 and new_pos != 0:
            self.entry_price = current_price
        
        self.position = new_pos
        self.current_step += 1
        
        # Update PnL
        next_price = self._get_price()
        if self.position != 0:
            price_change = (next_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance
            self.time_in_position += 1
        else:
            self.unrealized_pnl = 0.0
            self.time_in_position = 0
        
        # Step return
        if old_position != 0:
            step_return = old_position * (next_price - current_price) / current_price
        else:
            step_return = 0.0
        
        self.returns_history.append(step_return)
        self.recent_returns = np.roll(self.recent_returns, -1)
        self.recent_returns[-1] = step_return
        
        if len(self.returns_history) > 10:
            self.recent_volatility = np.std(self.returns_history[-20:]) + 1e-8
        
        # Update max balance
        total_equity = self.balance + self.unrealized_pnl
        self.max_balance = max(self.max_balance, total_equity)
        
        # Termination
        drawdown = (self.max_balance - total_equity) / self.max_balance if self.max_balance > 0 else 0
        terminated = (total_equity <= 0 or drawdown > self.config.max_drawdown or 
                      self.data_start_idx + 100 + self.current_step >= len(self.price_data) - 1)
        truncated = self.current_step >= self.config.max_steps
        
        info = self._get_info()
        info["step_return"] = step_return
        info["drawdown"] = drawdown
        
        return self._get_observation(), step_return, terminated, truncated, info
    
    def _get_price(self):
        idx = min(self.data_start_idx + 100 + self.current_step, len(self.price_data) - 1)
        return float(self.price_data[idx])
    
    def _get_observation(self):
        # TFT context (zeros if no model)
        context = np.zeros(self.config.context_dim, dtype=np.float32)
        
        # Account state
        total_equity = self.balance + self.unrealized_pnl
        drawdown = (self.max_balance - total_equity) / self.max_balance if self.max_balance > 0 else 0
        
        account = np.array([
            self.balance / 10000,
            self.position,
            self.unrealized_pnl / 10000,
            self.time_in_position / 100,
            drawdown,
            *self.recent_returns,
            self.recent_volatility * 100,
        ], dtype=np.float32)
        
        return np.concatenate([context, account])
    
    def _get_info(self):
        total_equity = self.balance + self.unrealized_pnl
        return {
            "balance": self.balance,
            "position": self.position,
            "total_equity": total_equity,
            "total_return": (total_equity - self.config.initial_balance) / self.config.initial_balance,
        }
    
    def get_sharpe_ratio(self):
        if len(self.returns_history) < 2:
            return 0.0
        returns = np.array(self.returns_history)
        return (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252 * 24)


# ============================================================================
# CELL 5: DIFFERENTIAL SHARPE RATIO REWARD
# ============================================================================

class DifferentialSharpeRatio:
    """DSR reward function - penalizes volatility, rewards risk-adjusted returns."""
    
    def __init__(self, eta: float = 0.01, drawdown_penalty: float = 1.0, volatility_penalty: float = 0.5):
        self.eta = eta
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.A = 0.0  # EMA of returns
        self.B = 0.0001  # EMA of squared returns
        self.step_count = 0
    
    def reset(self):
        self.A = 0.0
        self.B = 0.0001
        self.step_count = 0
    
    def compute(self, step_return: float, position: float = 0.0, drawdown: float = 0.0) -> float:
        self.step_count += 1
        
        delta_A = step_return - self.A
        delta_B = step_return**2 - self.B
        
        denom = (self.B - self.A**2) ** 1.5
        if abs(denom) < 1e-8:
            dsr = step_return * 100
        else:
            dsr = (self.A * delta_B - 0.5 * self.B * delta_A) / denom
        
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        
        # Penalties
        reward = dsr
        if drawdown > 0:
            reward -= self.drawdown_penalty * drawdown**2
        if self.B > self.A**2:
            reward -= self.volatility_penalty * np.sqrt(self.B - self.A**2)
        
        return float(np.clip(reward, -10, 10))


# ============================================================================
# CELL 6: POPART NORMALIZATION
# ============================================================================

class PopArtNormalizer(nn.Module):
    """Adaptive reward normalization for handling crypto volatility."""
    
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
# CELL 7: QR-DQN NETWORK
# ============================================================================

class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN - models full return distribution."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int], num_actions: int = 3, 
                 num_quantiles: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        # Feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        
        # Output with PopArt
        self.popart = PopArtNormalizer(in_dim, num_actions * num_quantiles)
        
        # Quantile midpoints
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
# CELL 8: REPLAY BUFFER
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
        return (
            self.states[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.dones[idx].to(self.device),
        )
    
    def __len__(self):
        return self.size


# ============================================================================
# CELL 9: QR-DQN AGENT
# ============================================================================

class QRDQNAgent:
    def __init__(self, state_dim: int, hidden_dims: List[int], num_quantiles: int = 32,
                 lr: float = 3e-4, gamma: float = 0.99, batch_size: int = 256,
                 buffer_size: int = 100000, min_buffer: int = 10000,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000, device: str = "cuda"):
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.num_quantiles = num_quantiles
        
        # Networks
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
        
        n_params = sum(p.numel() for p in self.online.parameters())
        logger.info(f"QR-DQN Agent: {n_params:,} parameters")
    
    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online.get_action(state_t, self.epsilon)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.buffer) < self.min_buffer:
            return {"loss": 0.0}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Current Q
        current_q = self.online(states)
        current_q = current_q.gather(1, actions.unsqueeze(-1).expand(-1, 1, self.num_quantiles)).squeeze(1)
        
        # Target Q (Double DQN)
        with torch.no_grad():
            next_actions = self.online.get_q_values(next_states).argmax(dim=-1, keepdim=True)
            next_q = self.target(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(-1).expand(-1, 1, self.num_quantiles)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Quantile Huber loss
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
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
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


# ============================================================================
# CELL 10: TRAINING LOOP
# ============================================================================

def generate_price_data(n_samples: int = 100000) -> np.ndarray:
    """Generate synthetic GARCH-like price data. (Deprecated: use generate_realistic_price_data)"""
    return generate_realistic_price_data(n_samples)


def fetch_binance_data(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 365,
) -> Optional[np.ndarray]:
    """Fetch real price data from Binance.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
        interval: Candle interval (1m, 5m, 1h, 4h, 1d)
        days: Number of days of historical data
        
    Returns:
        Array of close prices or None if fetch fails
    """
    try:
        import ccxt
    except ImportError:
        logger.warning("ccxt not installed - install with: pip install ccxt")
        return None
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        logger.info(f"Fetching {days} days of {symbol} data from Binance...")
        
        # Convert interval to milliseconds
        interval_ms = {
            '1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000,
            '4h': 14400000, '1d': 86400000
        }.get(interval, 3600000)
        
        all_ohlcv = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 3600 * 1000)
        
        current_time = start_time
        batch_count = 0
        while current_time < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, interval, current_time, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_time = ohlcv[-1][0] + interval_ms
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"  Fetched {len(all_ohlcv):,} candles...")
                    
            except Exception as e:
                logger.warning(f"Error fetching batch: {e}")
                break
        
        if not all_ohlcv:
            logger.warning("No data fetched")
            return None
        
        # Extract close prices and sort by timestamp
        prices = np.array([candle[4] for candle in sorted(all_ohlcv, key=lambda x: x[0])], dtype=np.float32)
        logger.info(f"✅ Fetched {len(prices):,} candles for {symbol}")
        
        return prices
        
    except Exception as e:
        logger.error(f"Error fetching Binance data: {e}")
        return None


def fetch_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    days: int = 365,
) -> Optional[np.ndarray]:
    """Fetch real price data from Bybit (alternative to Binance).
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval in minutes (1, 5, 15, 60, 240, 1440)
        days: Number of days of historical data
        
    Returns:
        Array of close prices or None if fetch fails
    """
    try:
        import ccxt
    except ImportError:
        logger.warning("ccxt not installed")
        return None
    
    try:
        exchange = ccxt.bybit({'enableRateLimit': True})
        logger.info(f"Fetching {days} days of {symbol} data from Bybit...")
        
        all_ohlcv = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 3600 * 1000)
        
        current_time = start_time
        batch_count = 0
        while current_time < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, interval, current_time, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_time = ohlcv[-1][0] + int(interval) * 60000
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"  Fetched {len(all_ohlcv):,} candles...")
                    
            except Exception as e:
                logger.warning(f"Error fetching batch: {e}")
                break
        
        if not all_ohlcv:
            logger.warning("No data fetched from Bybit")
            return None
        
        prices = np.array([candle[4] for candle in sorted(all_ohlcv, key=lambda x: x[0])], dtype=np.float32)
        logger.info(f"✅ Fetched {len(prices):,} candles for {symbol} from Bybit")
        
        return prices
        
    except Exception as e:
        logger.error(f"Error fetching Bybit data: {e}")
        return None


def generate_realistic_price_data(n_samples: int = 100000, trend: float = 0.00001) -> np.ndarray:
    """Generate realistic price data with trend and mean reversion (alternative when API blocked).
    
    Args:
        n_samples: Number of price points to generate
        trend: Daily drift (0.00001 = 0.001% per hour)
        
    Returns:
        Array of simulated prices
    """
    np.random.seed(42)
    
    # GARCH-like volatility
    volatility = np.zeros(n_samples)
    volatility[0] = 0.001
    for i in range(1, n_samples):
        volatility[i] = 0.00001 + 0.1 * volatility[i-1] + 0.85 * (np.random.randn()**2) * 0.0001
    
    # Mean reversion component
    mean_price = 50000
    mean_reversion = 0.01
    price_deviation = 0.0
    
    prices = [mean_price]
    for i in range(1, n_samples):
        # Mean reversion
        price_deviation = price_deviation * (1 - mean_reversion) + np.random.randn() * 0.001
        
        # Log returns with trend
        log_return = (trend + price_deviation) + np.random.randn() * np.sqrt(volatility[i])
        new_price = prices[-1] * np.exp(log_return)
        prices.append(new_price)
    
    return np.array(prices, dtype=np.float32)


def train_rl_agent(total_timesteps: int = 100000, eval_freq: int = 5000, use_real_data: bool = False,
                   symbol: str = "BTCUSDT", days: int = 365):
    """Main training function.
    
    Args:
        total_timesteps: Number of training steps
        eval_freq: Evaluation frequency
        use_real_data: If True, try to fetch real data; if False, use synthetic data
        symbol: Trading symbol (e.g., BTCUSDT, ETHUSDT)
        days: Number of days of historical data to fetch
    """
    
    # Get GPU settings
    settings = gpu_info["settings"]
    
    # Reduce min_buffer for faster start
    min_buffer = min(1000, settings["buffer_size"] // 10)
    
    # Generate or fetch data
    if use_real_data:
        logger.info("Attempting to fetch real data...")
        
        # Try Binance first
        prices = fetch_binance_data(symbol, interval="1h", days=days)
        
        # If Binance blocked, try Bybit
        if prices is None:
            logger.warning("Binance access failed, trying Bybit...")
            prices = fetch_bybit_data(symbol, interval="60", days=days)
        
        # If both fail, use realistic synthetic
        if prices is None:
            logger.warning("API access restricted, using realistic synthetic data")
            logger.info("💡 Tip: Use VPN to access Binance from restricted regions")
            prices = generate_realistic_price_data(200000, trend=0.00001)
        else:
            # Pad with synthetic if fetched data is too small
            if len(prices) < 10000:
                logger.warning(f"Limited data fetched ({len(prices)} candles), padding with synthetic...")
                additional = generate_realistic_price_data(200000 - len(prices))
                prices = np.concatenate([prices, additional])
    else:
        logger.info("Generating synthetic price data...")
        prices = generate_realistic_price_data(200000)
    
    logger.info(f"✅ Using {len(prices):,} price points for training")
    
    # Create env and agent
    env_config = EnvConfig(context_dim=64, max_steps=1000)
    env = TradingEnv(env_config, prices)
    
    agent = QRDQNAgent(
        state_dim=80,
        hidden_dims=settings["hidden_dims"],
        num_quantiles=settings["num_quantiles"],
        batch_size=settings["batch_size"],
        buffer_size=settings["buffer_size"],
        min_buffer=min_buffer,  # Start training sooner
        device=DEVICE,
    )
    logger.info(f"✅ Agent created, will start training after {min_buffer:,} samples")
    
    dsr = DifferentialSharpeRatio()
    
    # Training
    logger.info("=" * 60)
    logger.info("STARTING RL TRAINING")
    logger.info(f"Timesteps: {total_timesteps:,} | Batch: {settings['batch_size']} | Buffer: {settings['buffer_size']:,}")
    logger.info("=" * 60)
    
    state, _ = env.reset()
    dsr.reset()
    episode = 0
    episode_reward = 0
    episode_returns = []
    
    start_time = time.time()
    
    for step in range(total_timesteps):
        action = agent.select_action(state)
        next_state, raw_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # DSR reward
        reward = dsr.compute(info.get("step_return", raw_reward), env.position, info.get("drawdown", 0))
        agent.store(state, action, reward, next_state, done)
        
        metrics = agent.train_step()
        
        state = next_state
        episode_reward += reward
        
        if done:
            episode += 1
            episode_returns.append(info.get("total_return", 0))
            
            state, _ = env.reset()
            dsr.reset()
            episode_reward = 0
        
        # Progress bar every 1000 steps
        if (step + 1) % 1000 == 0:
            pct = (step + 1) / total_timesteps * 100
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"\r[{bar}] {pct:.1f}% - Step {step+1:,}", end="", flush=True)
        
        # Detailed logging
        if (step + 1) % eval_freq == 0:
            print()  # New line after progress bar
            elapsed = time.time() - start_time
            mean_return = np.mean(episode_returns[-10:]) if episode_returns else 0
            logger.info(
                f"Step {step+1:,}/{total_timesteps:,} | "
                f"Episodes: {episode} | "
                f"Return: {mean_return*100:+.2f}% | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Time: {elapsed/60:.1f}m"
            )
    
    # Save
    agent.save("best_rl_agent.pt")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to best_rl_agent.pt")
    logger.info("=" * 60)
    
    return agent


# ============================================================================
# CELL 11: RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Training options
    use_real_data = "--real" in sys.argv
    symbol = "BTCUSDT"  # Change to ETHUSDT, BNBUSDT, etc. for other assets
    
    if use_real_data:
        print("\n" + "="*60)
        print("TRAINING WITH REAL DATA FROM BINANCE")
        print("="*60 + "\n")
        agent = train_rl_agent(
            total_timesteps=100000,
            eval_freq=10000,
            use_real_data=True,
            symbol=symbol,
            days=365
        )
    else:
        print("\n" + "="*60)
        print("TRAINING WITH SYNTHETIC DATA")
        print("To use real Binance data, run with: python colab_train_rl.py --real")
        print("="*60 + "\n")
        agent = train_rl_agent(
            total_timesteps=100000,
            eval_freq=10000,
            use_real_data=False
        )
    
    print("\n✅ Training complete!")
    print("📁 Model saved to: best_rl_agent.pt")
    print("\nNext steps:")
    print("  1. Run backtesting: python colab_backtest_rl.py")
    print("  2. Increase timesteps for better performance")
    print("  3. Try different symbols: ETHUSDT, BNBUSDT, etc.")
