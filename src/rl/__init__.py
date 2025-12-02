"""
Reinforcement Learning module for crypto trading.

Phase 3: The "Brain" - RL Agent with QR-DQN and DSR reward.
"""

from src.rl.environment import TradingEnv, EnvConfig
from src.rl.rewards import DifferentialSharpeRatio, RewardConfig
from src.rl.agent import QRDQNAgent, QRDQNConfig
from src.rl.popart import PopArtNormalizer
from src.rl.trainer import RLTrainer, RLTrainingConfig

__all__ = [
    # Environment
    "TradingEnv",
    "EnvConfig",
    # Rewards
    "DifferentialSharpeRatio",
    "RewardConfig",
    # Agent
    "QRDQNAgent",
    "QRDQNConfig",
    # Normalization
    "PopArtNormalizer",
    # Training
    "RLTrainer",
    "RLTrainingConfig",
]
