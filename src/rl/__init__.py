"""
Reinforcement Learning module for crypto trading.

Phase 3: The "Brain" - RL Agent with QR-DQN and DSR reward.
"""

from src.rl.environment import TradingEnv, EnvConfig
from src.rl.rewards import DifferentialSharpeRatio, RewardConfig
from src.rl.agent import QRDQNAgent, QRDQNConfig
from src.rl.popart import PopArtNormalizer
from src.rl.trainer import RLTrainer, RLTrainingConfig
from src.rl.backtester import Backtester, BacktestConfig, BacktestMetrics, run_backtest
from src.rl.data_loader import get_price_data, get_statistics, split_data, generate_synthetic_data

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
    # Backtesting
    "Backtester",
    "BacktestConfig",
    "BacktestMetrics",
    "run_backtest",
    # Data
    "get_price_data",
    "get_statistics",
    "split_data",
    "generate_synthetic_data",
]
