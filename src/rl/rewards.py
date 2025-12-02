"""
Reward Functions for Reinforcement Learning.

Implements:
1. Differential Sharpe Ratio (DSR) - Risk-adjusted returns
2. PopArt Normalization - Adaptive reward scaling
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    
    # DSR parameters
    eta: float = 0.01  # Decay factor for exponential moving averages
    risk_free_rate: float = 0.0  # Annual risk-free rate
    
    # Risk penalty
    max_drawdown_penalty: float = 1.0  # Penalty multiplier for drawdowns
    volatility_penalty: float = 0.5  # Penalty for high volatility
    
    # Reward clipping
    clip_reward: bool = True
    reward_clip_value: float = 10.0
    
    # Additional penalties
    holding_penalty: float = 0.0001  # Small penalty for holding (encourage trading)
    transaction_penalty: float = 0.001  # Additional penalty per trade


class DifferentialSharpeRatio:
    """
    Differential Sharpe Ratio (DSR) reward function.
    
    From: "Reinforcement Learning for Trading" (Moody & Saffell, 2001)
    
    DSR provides a reward that:
    1. Rewards risk-adjusted returns (not just raw returns)
    2. Penalizes volatility
    3. Uses exponential moving averages for stability
    
    Formula:
        DSR_t = (A_{t-1} * ΔB_t - 0.5 * B_{t-1} * ΔA_t) / (B_{t-1} - A_{t-1}^2)^{1.5}
    
    Where:
        A_t = EMA of returns
        B_t = EMA of squared returns
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # Exponential moving averages
        self.A: float = 0.0  # EMA of returns
        self.B: float = 0.0001  # EMA of squared returns (initialized small to avoid div by zero)
        
        # Tracking
        self.step_count: int = 0
        self.reward_history: List[float] = []
        
    def reset(self):
        """Reset the DSR state for a new episode."""
        self.A = 0.0
        self.B = 0.0001
        self.step_count = 0
        self.reward_history = []
    
    def compute(
        self,
        step_return: float,
        position: float = 0.0,
        drawdown: float = 0.0,
        action_changed: bool = False,
    ) -> float:
        """
        Compute the Differential Sharpe Ratio reward.
        
        Args:
            step_return: The raw return for this step (R_t)
            position: Current position (-1 to 1)
            drawdown: Current drawdown (0 to 1)
            action_changed: Whether action changed from previous step
        
        Returns:
            DSR reward value
        """
        eta = self.config.eta
        R_t = step_return
        
        # Update step count
        self.step_count += 1
        
        # Compute deltas
        delta_A = R_t - self.A
        delta_B = R_t**2 - self.B
        
        # Compute DSR
        # DSR = (A * ΔB - 0.5 * B * ΔA) / (B - A^2)^1.5
        denominator = (self.B - self.A**2) ** 1.5
        
        if abs(denominator) < 1e-8:
            # Avoid division by zero at start
            dsr = R_t * 100  # Scale up small returns
        else:
            numerator = self.A * delta_B - 0.5 * self.B * delta_A
            dsr = numerator / denominator
        
        # Update EMAs
        self.A = self.A + eta * delta_A
        self.B = self.B + eta * delta_B
        
        # Apply penalties
        reward = dsr
        
        # Drawdown penalty (penalize being in drawdown)
        if drawdown > 0:
            drawdown_penalty = self.config.max_drawdown_penalty * drawdown**2
            reward -= drawdown_penalty
        
        # Volatility penalty (based on recent variance)
        if self.B > self.A**2:
            volatility = np.sqrt(self.B - self.A**2)
            vol_penalty = self.config.volatility_penalty * volatility
            reward -= vol_penalty
        
        # Holding penalty (encourage activity)
        if position == 0:
            reward -= self.config.holding_penalty
        
        # Transaction penalty
        if action_changed:
            reward -= self.config.transaction_penalty
        
        # Clip reward
        if self.config.clip_reward:
            reward = np.clip(reward, -self.config.reward_clip_value, self.config.reward_clip_value)
        
        self.reward_history.append(reward)
        
        return float(reward)
    
    def get_stats(self) -> dict:
        """Get statistics about the reward function."""
        return {
            "A": self.A,
            "B": self.B,
            "sharpe_estimate": self.A / np.sqrt(self.B - self.A**2 + 1e-8),
            "step_count": self.step_count,
            "mean_reward": np.mean(self.reward_history) if self.reward_history else 0,
            "std_reward": np.std(self.reward_history) if self.reward_history else 0,
        }


class AsymmetricDSR(DifferentialSharpeRatio):
    """
    Asymmetric DSR that penalizes losses more than it rewards gains.
    
    Good for crypto where avoiding large losses is crucial.
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        loss_multiplier: float = 2.0,
    ):
        super().__init__(config)
        self.loss_multiplier = loss_multiplier
    
    def compute(
        self,
        step_return: float,
        position: float = 0.0,
        drawdown: float = 0.0,
        action_changed: bool = False,
    ) -> float:
        """Compute asymmetric DSR reward."""
        
        # Apply asymmetric scaling to returns before DSR
        if step_return < 0:
            scaled_return = step_return * self.loss_multiplier
        else:
            scaled_return = step_return
        
        return super().compute(scaled_return, position, drawdown, action_changed)


class ProfitReward:
    """
    Simple profit-based reward for comparison.
    
    R_t = position * price_return - transaction_cost
    """
    
    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost
        self.reward_history: List[float] = []
    
    def reset(self):
        self.reward_history = []
    
    def compute(
        self,
        step_return: float,
        position: float = 0.0,
        drawdown: float = 0.0,
        action_changed: bool = False,
    ) -> float:
        """Compute simple profit reward."""
        reward = step_return
        
        if action_changed:
            reward -= self.transaction_cost
        
        self.reward_history.append(reward)
        return reward


class RewardWrapper:
    """
    Wrapper that combines multiple reward functions.
    """
    
    def __init__(
        self,
        primary: DifferentialSharpeRatio,
        auxiliary: Optional[List] = None,
        weights: Optional[List[float]] = None,
    ):
        self.primary = primary
        self.auxiliary = auxiliary or []
        self.weights = weights or [0.1] * len(self.auxiliary)
    
    def reset(self):
        self.primary.reset()
        for aux in self.auxiliary:
            aux.reset()
    
    def compute(
        self,
        step_return: float,
        position: float = 0.0,
        drawdown: float = 0.0,
        action_changed: bool = False,
    ) -> float:
        """Compute combined reward."""
        reward = self.primary.compute(step_return, position, drawdown, action_changed)
        
        for aux, weight in zip(self.auxiliary, self.weights):
            reward += weight * aux.compute(step_return, position, drawdown, action_changed)
        
        return reward


def create_reward_function(
    reward_type: str = "dsr",
    **kwargs,
) -> DifferentialSharpeRatio:
    """
    Factory function to create reward functions.
    
    Args:
        reward_type: "dsr", "asymmetric_dsr", or "profit"
        **kwargs: Additional arguments for RewardConfig
    
    Returns:
        Reward function instance
    """
    config = RewardConfig(**kwargs)
    
    if reward_type == "dsr":
        return DifferentialSharpeRatio(config)
    elif reward_type == "asymmetric_dsr":
        return AsymmetricDSR(config, loss_multiplier=kwargs.get("loss_multiplier", 2.0))
    elif reward_type == "profit":
        return ProfitReward(transaction_cost=kwargs.get("transaction_penalty", 0.001))
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
