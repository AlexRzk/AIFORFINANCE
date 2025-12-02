"""
Train QR-DQN Agent - Phase 3 Training Script.

This script:
1. Loads pretrained TFT model (optional)
2. Creates trading environment
3. Trains QR-DQN agent with DSR reward
4. Saves checkpoints

Usage:
    python scripts/train_rl.py --timesteps 500000
    
For Colab:
    !python scripts/train_rl.py --colab
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.environment import TradingEnv, EnvConfig
from src.rl.rewards import DifferentialSharpeRatio, RewardConfig
from src.rl.agent import QRDQNAgent, QRDQNConfig
from src.rl.trainer import RLTrainer, RLTrainingConfig, get_gpu_optimal_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_price_data(n_samples: int = 100000) -> np.ndarray:
    """
    Generate synthetic price data for training.
    
    Uses a GARCH-like volatility process for realistic behavior.
    """
    np.random.seed(42)
    
    # Volatility clustering (GARCH-like)
    volatility = np.zeros(n_samples)
    volatility[0] = 0.001
    
    for i in range(1, n_samples):
        shock = np.random.randn() ** 2
        volatility[i] = 0.00001 + 0.1 * volatility[i-1] + 0.85 * shock * 0.0001
    
    # Generate returns with volatility clustering
    returns = np.random.randn(n_samples) * np.sqrt(volatility) + 0.00001
    
    # Convert to prices
    log_prices = np.cumsum(returns) + np.log(50000)
    prices = np.exp(log_prices)
    
    logger.info(f"Generated {n_samples} price samples")
    logger.info(f"Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")
    
    return prices


def main(args):
    """Main training function."""
    
    # =========================================================================
    # GPU Detection
    # =========================================================================
    logger.info("=" * 60)
    logger.info("GPU INFORMATION")
    logger.info("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Memory: {gpu_memory:.1f} GB")
        
        # Get optimal settings
        optimal = get_gpu_optimal_config(gpu_name)
        logger.info(f"Optimal settings: {optimal}")
        
        if args.auto_config:
            args.batch_size = optimal.get("batch_size", args.batch_size)
            args.hidden_dims = optimal.get("hidden_dims", args.hidden_dims)
            args.num_quantiles = optimal.get("num_quantiles", args.num_quantiles)
            args.buffer_size = optimal.get("buffer_size", args.buffer_size)
    else:
        logger.info("No GPU available, using CPU")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # =========================================================================
    # Load TFT Model (if provided)
    # =========================================================================
    tft_model = None
    feature_data = None
    context_dim = args.context_dim
    
    if args.tft_checkpoint and Path(args.tft_checkpoint).exists():
        logger.info(f"Loading TFT model from {args.tft_checkpoint}")
        
        from src.models.tft import TFTConfig, TemporalFusionTransformer
        
        checkpoint = torch.load(args.tft_checkpoint, map_location=device, weights_only=False)
        
        if "config" in checkpoint:
            tft_config = checkpoint["config"]
        else:
            tft_config = TFTConfig()
        
        tft_model = TemporalFusionTransformer(tft_config)
        tft_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        tft_model = tft_model.to(device)
        tft_model.eval()
        
        context_dim = tft_config.embed_dim
        logger.info(f"Loaded TFT model with context dim: {context_dim}")
    
    # =========================================================================
    # Generate or Load Price Data
    # =========================================================================
    if args.data_path and Path(args.data_path).exists():
        import pandas as pd
        logger.info(f"Loading price data from {args.data_path}")
        df = pd.read_parquet(args.data_path)
        price_data = df["mid_price"].values if "mid_price" in df.columns else df["price"].values
        
        if tft_model is not None:
            feature_cols = [c for c in df.columns if c.startswith(("price_", "ofi_", "volume_", "fdi", "volatility", "spread"))]
            if feature_cols:
                feature_data = df[feature_cols].values.astype(np.float32)
    else:
        logger.info("Generating synthetic price data...")
        price_data = generate_synthetic_price_data(args.n_samples)
    
    # =========================================================================
    # Create Configurations
    # =========================================================================
    env_config = EnvConfig(
        context_dim=context_dim,
        initial_balance=args.initial_balance,
        max_position=args.max_position,
        trading_fee=args.trading_fee,
        max_steps=args.max_steps,
        max_drawdown=args.max_drawdown,
    )
    
    # State dim = context + account state (16)
    state_dim = context_dim + 16
    
    agent_config = QRDQNConfig(
        state_dim=state_dim,
        hidden_dims=args.hidden_dims,
        num_actions=3,
        num_quantiles=args.num_quantiles,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        use_popart=args.use_popart,
        device=device,
    )
    
    reward_config = RewardConfig(
        eta=args.dsr_eta,
        max_drawdown_penalty=args.drawdown_penalty,
        volatility_penalty=args.volatility_penalty,
    )
    
    training_config = RLTrainingConfig(
        total_timesteps=args.timesteps,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        save_frequency=args.save_frequency,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        env_config=env_config,
        agent_config=agent_config,
        reward_config=reward_config,
        device=device,
    )
    
    # =========================================================================
    # Log Configuration
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Context dim: {context_dim}")
    logger.info(f"State dim: {state_dim}")
    logger.info(f"Initial balance: ${env_config.initial_balance:,.2f}")
    logger.info(f"Max position: {env_config.max_position}")
    logger.info(f"Trading fee: {env_config.trading_fee*100:.2f}%")
    logger.info(f"Max drawdown: {env_config.max_drawdown*100:.0f}%")
    
    logger.info("=" * 60)
    logger.info("AGENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Hidden dims: {agent_config.hidden_dims}")
    logger.info(f"Num quantiles: {agent_config.num_quantiles}")
    logger.info(f"Batch size: {agent_config.batch_size}")
    logger.info(f"Buffer size: {agent_config.buffer_size:,}")
    logger.info(f"Learning rate: {agent_config.learning_rate}")
    logger.info(f"PopArt: {agent_config.use_popart}")
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Total timesteps: {training_config.total_timesteps:,}")
    logger.info(f"Eval frequency: {training_config.eval_frequency:,}")
    logger.info(f"Device: {device}")
    
    # =========================================================================
    # Create Trainer and Train
    # =========================================================================
    trainer = RLTrainer(
        config=training_config,
        tft_model=tft_model,
        price_data=price_data,
        feature_data=feature_data,
    )
    
    # Log parameter count
    n_params = sum(p.numel() for p in trainer.agent.online_net.parameters())
    logger.info(f"Agent parameters: {n_params:,}")
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    stats = trainer.train()
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total steps: {stats['total_steps']:,}")
    logger.info(f"Episodes: {stats['episodes']}")
    logger.info(f"Best eval return: {stats['best_eval_return']:.4f}")
    logger.info(f"Time: {stats['elapsed_time']/3600:.2f} hours")
    
    return trainer.agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QR-DQN Agent")
    
    # Data
    parser.add_argument("--data-path", type=str, default=None, help="Path to price data")
    parser.add_argument("--n-samples", type=int, default=100000, help="Synthetic samples")
    parser.add_argument("--tft-checkpoint", type=str, default=None, help="Path to TFT model")
    
    # Environment
    parser.add_argument("--context-dim", type=int, default=64, help="TFT context dimension")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--max-position", type=float, default=1.0, help="Max position")
    parser.add_argument("--trading-fee", type=float, default=0.0004, help="Trading fee")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--max-drawdown", type=float, default=0.2, help="Max drawdown")
    
    # Agent
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--num-quantiles", type=int, default=32, help="Number of quantiles")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--min-buffer-size", type=int, default=10000, help="Min buffer before training")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000, help="Epsilon decay steps")
    parser.add_argument("--use-popart", action="store_true", default=True, help="Use PopArt")
    parser.add_argument("--no-popart", action="store_false", dest="use_popart")
    
    # Reward
    parser.add_argument("--dsr-eta", type=float, default=0.01, help="DSR decay rate")
    parser.add_argument("--drawdown-penalty", type=float, default=1.0, help="Drawdown penalty")
    parser.add_argument("--volatility-penalty", type=float, default=0.5, help="Volatility penalty")
    
    # Training
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--eval-frequency", type=int, default=10000, help="Eval frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Eval episodes")
    parser.add_argument("--save-frequency", type=int, default=50000, help="Checkpoint frequency")
    
    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/rl")
    parser.add_argument("--log-dir", type=str, default="logs/rl")
    
    # Auto-config
    parser.add_argument("--auto-config", action="store_true", help="Auto-configure for GPU")
    parser.add_argument("--colab", action="store_true", help="Colab mode")
    
    args = parser.parse_args()
    
    if args.colab:
        args.auto_config = True
    
    main(args)
