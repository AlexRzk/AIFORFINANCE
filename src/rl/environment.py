"""
Custom Trading Environment for Reinforcement Learning.

Gymnasium-compatible environment that:
1. Uses TFT context vector as primary state
2. Tracks account balance and position
3. Supports discrete actions: Buy, Sell, Hold
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the trading environment."""
    
    # TFT context
    context_dim: int = 64  # Dimension of TFT context vector
    
    # Account settings
    initial_balance: float = 10000.0
    max_position: float = 1.0  # Maximum position size (1.0 = 100% of balance)
    position_step: float = 0.25  # Position change per action
    
    # Trading costs
    trading_fee: float = 0.0004  # 0.04% taker fee (Binance Futures)
    slippage: float = 0.0001  # 0.01% average slippage
    
    # Episode settings
    max_steps: int = 1000
    lookback_window: int = 100  # For TFT context
    
    # Reward settings
    reward_scaling: float = 1.0
    use_log_returns: bool = True
    
    # Risk management
    max_drawdown: float = 0.2  # 20% max drawdown before episode ends
    stop_loss: float = 0.05  # 5% stop loss per trade
    
    # State normalization
    normalize_balance: bool = True
    balance_scale: float = 10000.0


class TradingEnv(gym.Env):
    """
    Crypto Trading Environment.
    
    State Space:
        - TFT context vector (context_dim)
        - Account balance (normalized)
        - Current position (-1 to 1)
        - Unrealized PnL
        - Time in position
        - Recent returns (last 10)
        
    Action Space:
        - 0: Hold
        - 1: Buy (increase position)
        - 2: Sell (decrease position)
        
    Or continuous:
        - Target position from -1 (full short) to 1 (full long)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        config: EnvConfig,
        price_data: Optional[np.ndarray] = None,
        tft_model: Optional[torch.nn.Module] = None,
        feature_data: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Data
        self.price_data = price_data
        self.feature_data = feature_data
        self.tft_model = tft_model
        
        if tft_model is not None:
            tft_model.eval()
            for param in tft_model.parameters():
                param.requires_grad = False
        
        # State dimension: context + account state
        # Context (64) + balance (1) + position (1) + unrealized_pnl (1) + 
        # time_in_position (1) + drawdown (1) + recent_returns (10) + volatility (1) = 16
        self.account_state_dim = 16
        self.state_dim = config.context_dim + self.account_state_dim
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        
        # Discrete actions: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: "hold", 1: "buy", 2: "sell"}
        
        # Episode state
        self._reset_state()
        
        # History for rendering/analysis
        self.history: Dict[str, List] = {}
    
    def _reset_state(self):
        """Reset internal state variables."""
        self.balance = self.config.initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.time_in_position = 0
        self.current_step = 0
        self.max_balance = self.balance
        self.recent_returns = np.zeros(10)
        self.recent_volatility = 0.01
        
        # Track for DSR
        self.returns_history = []
        self.equity_curve = [self.balance]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self._reset_state()
        
        # Random start position in data
        if self.price_data is not None:
            max_start = len(self.price_data) - self.config.max_steps - self.config.lookback_window
            if max_start > 0:
                self.data_start_idx = self.np_random.integers(0, max_start)
            else:
                self.data_start_idx = 0
        else:
            self.data_start_idx = 0
        
        # Clear history
        self.history = {
            "balance": [],
            "position": [],
            "price": [],
            "action": [],
            "reward": [],
            "pnl": [],
        }
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current and next price
        current_price = self._get_current_price()
        
        # Execute action
        old_position = self.position
        self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        next_price = self._get_current_price()
        
        # Update unrealized PnL
        if self.position != 0:
            price_change = (next_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance
            self.time_in_position += 1
        else:
            self.unrealized_pnl = 0.0
            self.time_in_position = 0
        
        # Calculate step return
        step_return = self._calculate_step_return(old_position, current_price, next_price)
        self.returns_history.append(step_return)
        
        # Update recent returns
        self.recent_returns = np.roll(self.recent_returns, -1)
        self.recent_returns[-1] = step_return
        
        # Update volatility
        if len(self.returns_history) > 10:
            self.recent_volatility = np.std(self.returns_history[-20:]) + 1e-8
        
        # Update equity curve
        total_equity = self.balance + self.unrealized_pnl
        self.equity_curve.append(total_equity)
        self.max_balance = max(self.max_balance, total_equity)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_steps
        
        # Calculate reward (raw return, will be processed by DSR externally)
        reward = step_return * self.config.reward_scaling
        
        # Record history
        self.history["balance"].append(self.balance)
        self.history["position"].append(self.position)
        self.history["price"].append(next_price)
        self.history["action"].append(action)
        self.history["reward"].append(reward)
        self.history["pnl"].append(self.unrealized_pnl + self.realized_pnl)
        
        obs = self._get_observation()
        info = self._get_info()
        info["step_return"] = step_return
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, price: float):
        """Execute trading action."""
        position_change = 0.0
        
        if action == 1:  # Buy
            position_change = self.config.position_step
        elif action == 2:  # Sell
            position_change = -self.config.position_step
        
        new_position = np.clip(
            self.position + position_change,
            -self.config.max_position,
            self.config.max_position,
        )
        
        # Calculate trading cost
        position_delta = abs(new_position - self.position)
        if position_delta > 0:
            trade_value = position_delta * self.balance
            trading_cost = trade_value * (self.config.trading_fee + self.config.slippage)
            self.balance -= trading_cost
            self.realized_pnl -= trading_cost
        
        # Close existing position if changing direction
        if self.position != 0 and np.sign(new_position) != np.sign(self.position):
            # Realize PnL from closed position
            self.balance += self.unrealized_pnl
            self.realized_pnl += self.unrealized_pnl
            self.unrealized_pnl = 0.0
            self.entry_price = price
        elif self.position == 0 and new_position != 0:
            # New position
            self.entry_price = price
        
        self.position = new_position
    
    def _calculate_step_return(
        self, old_position: float, old_price: float, new_price: float
    ) -> float:
        """Calculate the return for this step."""
        if old_position == 0:
            return 0.0
        
        price_return = (new_price - old_price) / old_price
        position_return = old_position * price_return
        
        if self.config.use_log_returns:
            return np.log1p(position_return)
        return position_return
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        total_equity = self.balance + self.unrealized_pnl
        
        # Bankruptcy
        if total_equity <= 0:
            return True
        
        # Max drawdown exceeded
        drawdown = (self.max_balance - total_equity) / self.max_balance
        if drawdown > self.config.max_drawdown:
            return True
        
        # Stop loss hit on current position
        if self.position != 0 and self.entry_price > 0:
            position_pnl = self.unrealized_pnl / self.balance
            if position_pnl < -self.config.stop_loss:
                return True
        
        # Out of data
        if self.price_data is not None:
            data_idx = self.data_start_idx + self.config.lookback_window + self.current_step
            if data_idx >= len(self.price_data) - 1:
                return True
        
        return False
    
    def _get_current_price(self) -> float:
        """Get current price from data or generate synthetic."""
        if self.price_data is not None:
            idx = self.data_start_idx + self.config.lookback_window + self.current_step
            idx = min(idx, len(self.price_data) - 1)
            return float(self.price_data[idx])
        else:
            # Synthetic price for testing
            return 50000.0 * np.exp(0.0001 * self.current_step + 0.01 * np.random.randn())
    
    def _get_tft_context(self) -> np.ndarray:
        """Get context vector from TFT model."""
        if self.tft_model is not None and self.feature_data is not None:
            idx = self.data_start_idx + self.current_step
            end_idx = idx + self.config.lookback_window
            
            if end_idx <= len(self.feature_data):
                features = self.feature_data[idx:end_idx]
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    device = next(self.tft_model.parameters()).device
                    features_tensor = features_tensor.to(device)
                    context = self.tft_model.get_context_vector(features_tensor)
                    return context.cpu().numpy().flatten()
        
        # Return zeros if no TFT model
        return np.zeros(self.config.context_dim, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        # TFT context
        context = self._get_tft_context()
        
        # Normalize balance
        if self.config.normalize_balance:
            norm_balance = self.balance / self.config.balance_scale
        else:
            norm_balance = self.balance
        
        # Account state
        total_equity = self.balance + self.unrealized_pnl
        drawdown = (self.max_balance - total_equity) / self.max_balance if self.max_balance > 0 else 0
        
        account_state = np.array([
            norm_balance,                           # Normalized balance
            self.position,                          # Current position
            self.unrealized_pnl / self.config.balance_scale,  # Unrealized PnL
            self.time_in_position / 100.0,          # Time in position (normalized)
            drawdown,                               # Current drawdown
            *self.recent_returns,                   # Last 10 returns
            self.recent_volatility * 100,           # Recent volatility (scaled)
        ], dtype=np.float32)
        
        # Combine
        obs = np.concatenate([context, account_state])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        total_equity = self.balance + self.unrealized_pnl
        
        return {
            "balance": self.balance,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_equity": total_equity,
            "max_balance": self.max_balance,
            "drawdown": (self.max_balance - total_equity) / self.max_balance if self.max_balance > 0 else 0,
            "step": self.current_step,
            "total_return": (total_equity - self.config.initial_balance) / self.config.initial_balance,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step {self.current_step:4d} | "
                  f"Balance: ${info['total_equity']:,.2f} | "
                  f"Position: {self.position:+.2f} | "
                  f"Return: {info['total_return']*100:+.2f}%")
    
    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of the episode."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        
        # Annualized (assuming ~252 trading days, ~24 hours)
        annualization = np.sqrt(252 * 24)
        return (mean_return / std_return) * annualization
    
    def get_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) < 2:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns) + 1e-8
        annualization = np.sqrt(252 * 24)
        return (mean_return / downside_std) * annualization


def create_env(
    config: Optional[EnvConfig] = None,
    price_data: Optional[np.ndarray] = None,
    feature_data: Optional[np.ndarray] = None,
    tft_model: Optional[torch.nn.Module] = None,
    **kwargs,
) -> TradingEnv:
    """Factory function to create trading environment."""
    if config is None:
        config = EnvConfig(**kwargs)
    
    return TradingEnv(
        config=config,
        price_data=price_data,
        feature_data=feature_data,
        tft_model=tft_model,
    )
