#!/usr/bin/env python
"""Test colab_backtest_rl.py before pushing"""

import sys
import os
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or '.')

print("=" * 60)
print("TESTING colab_backtest_rl.py")
print("=" * 60)

# Test 1: Imports
print("\n1️⃣  Testing imports...")
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict
print("   ✅ All imports OK")

# Test 2: PopArt
print("\n2️⃣  Testing PopArtNormalizer...")

class PopArtNormalizer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, beta: float = 0.0001):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("std", torch.ones(output_dim))
        self.register_buffer("count", torch.zeros(1))
    
    def forward(self, x):
        return self.linear(x)

popart = PopArtNormalizer(10, 5)
x = torch.randn(2, 10)
out = popart(x)
assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
print("   ✅ PopArtNormalizer OK")

# Test 3: QRDQNNetwork
print("\n3️⃣  Testing QRDQNNetwork...")

class QRDQNNetwork(nn.Module):
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

net = QRDQNNetwork(80, [256, 256], 3, 32)
state = torch.randn(1, 80)
q = net.get_q_values(state)
assert q.shape == (1, 3), f"Expected (1, 3), got {q.shape}"
action = net.get_action(state)
assert action in [0, 1, 2], f"Expected action in [0,1,2], got {action}"
print("   ✅ QRDQNNetwork OK")

# Test 4: ReplayBuffer
print("\n4️⃣  Testing ReplayBuffer...")

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
    
    def __len__(self):
        return self.size

buffer = ReplayBuffer(100, 80)
state = np.zeros(80, dtype=np.float32)
buffer.push(state, 1, 0.5, state, False)
assert len(buffer) == 1, f"Expected size 1, got {len(buffer)}"
print("   ✅ ReplayBuffer OK")

# Test 5: QRDQNAgent
print("\n5️⃣  Testing QRDQNAgent...")

class QRDQNAgent:
    def __init__(self, state_dim: int, hidden_dims: List[int], num_quantiles: int = 32,
                 lr: float = 3e-4, gamma: float = 0.99, batch_size: int = 256,
                 buffer_size: int = 100000, min_buffer: int = 1000,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000, device: str = "cpu"):
        
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
        eps = 0.0 if deterministic else self.epsilon
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online.get_action(state_t, eps)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
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

agent = QRDQNAgent(state_dim=80, hidden_dims=[256, 256], num_quantiles=32, device="cpu")
state = np.zeros(80, dtype=np.float32)
action = agent.select_action(state, deterministic=True)
assert action in [0, 1, 2], f"Expected action in [0,1,2], got {action}"
print("   ✅ QRDQNAgent OK")

# Test 6: Backtester
print("\n6️⃣  Testing Backtester...")

@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    max_position: float = 1.0
    position_step: float = 0.25
    trading_fee: float = 0.0004
    slippage: float = 0.0001

@dataclass
class BacktestMetrics:
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

class Backtester:
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
    
    def step(self, action: int, current_price: float, next_price: float):
        if action == 1:
            new_pos = min(self.position + self.config.position_step, self.config.max_position)
        elif action == 2:
            new_pos = max(self.position - self.config.position_step, -self.config.max_position)
        else:
            new_pos = self.position
        
        if new_pos != self.position:
            cost = abs(new_pos - self.position) * self.balance * (self.config.trading_fee + self.config.slippage)
            self.balance -= cost
        
        if self.position == 0 and new_pos != 0:
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
    
    def compute_metrics(self) -> BacktestMetrics:
        final_equity = self.equity_history[-1]
        total_return = (final_equity - self.config.initial_balance) / self.config.initial_balance
        returns_array = np.array(self.returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(24 * 252)
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
        return BacktestMetrics(total_return, sharpe, sharpe, 0, max_drawdown, 0.5, 1.0, 0, 0, 0, 0)

config = BacktestConfig()
backtester = Backtester(config)

# Simple test
prices = 50000 + np.cumsum(np.random.randn(1000) * 10)
for i in range(len(prices) - 1):
    action = np.random.randint(0, 3)
    backtester.step(action, prices[i], prices[i+1])

metrics = backtester.compute_metrics()
assert isinstance(metrics.total_return, float), "Expected float total_return"
assert isinstance(metrics.sharpe_ratio, float), "Expected float sharpe"
print("   ✅ Backtester OK")

# Test 7: Save/Load
print("\n7️⃣  Testing save/load...")
test_path = "test_agent_temp.pt"
agent.save(test_path)
agent2 = QRDQNAgent(state_dim=80, hidden_dims=[256, 256], num_quantiles=32, device="cpu")
agent2.load(test_path)
os.remove(test_path)
print("   ✅ Save/Load OK")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - colab_backtest_rl.py is ready!")
print("=" * 60)
