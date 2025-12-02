"""
RL Training Infrastructure.

Supports:
1. GPU training (T4, A6000, H100)
2. Mixed precision training
3. TensorBoard logging
4. Checkpoint saving
5. Colab compatibility
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.environment import TradingEnv, EnvConfig
from src.rl.rewards import DifferentialSharpeRatio, RewardConfig
from src.rl.agent import QRDQNAgent, QRDQNConfig

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""
    
    # Training
    total_timesteps: int = 500000
    eval_frequency: int = 10000  # Steps between evaluations
    eval_episodes: int = 10
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints/rl"
    save_frequency: int = 50000
    
    # Logging
    log_dir: str = "logs/rl"
    log_frequency: int = 1000
    
    # Environment
    env_config: EnvConfig = field(default_factory=EnvConfig)
    
    # Agent
    agent_config: QRDQNConfig = field(default_factory=QRDQNConfig)
    
    # Reward
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def get_gpu_optimal_config(gpu_name: str) -> Dict:
    """Get optimal config for specific GPU."""
    gpu_lower = gpu_name.lower()
    
    if "t4" in gpu_lower:
        return {
            "batch_size": 128,
            "buffer_size": 50000,
            "hidden_dims": [256, 256],
            "num_quantiles": 32,
        }
    elif "a100" in gpu_lower or "a6000" in gpu_lower:
        return {
            "batch_size": 512,
            "buffer_size": 200000,
            "hidden_dims": [512, 512, 256],
            "num_quantiles": 64,
        }
    elif "h100" in gpu_lower:
        return {
            "batch_size": 1024,
            "buffer_size": 500000,
            "hidden_dims": [512, 512, 512],
            "num_quantiles": 64,
        }
    elif "4090" in gpu_lower or "3090" in gpu_lower:
        return {
            "batch_size": 512,
            "buffer_size": 200000,
            "hidden_dims": [512, 512, 256],
            "num_quantiles": 64,
        }
    else:
        # Default
        return {
            "batch_size": 256,
            "buffer_size": 100000,
            "hidden_dims": [256, 256, 128],
            "num_quantiles": 32,
        }


class RLTrainer:
    """
    Reinforcement Learning Trainer.
    
    Orchestrates:
    1. Environment interaction
    2. Agent training
    3. Logging and checkpointing
    """
    
    def __init__(
        self,
        config: RLTrainingConfig,
        env: Optional[TradingEnv] = None,
        agent: Optional[QRDQNAgent] = None,
        tft_model: Optional[torch.nn.Module] = None,
        price_data: Optional[np.ndarray] = None,
        feature_data: Optional[np.ndarray] = None,
    ):
        self.config = config
        
        # Setup device
        self.device = get_device(config.device)
        logger.info(f"Using device: {self.device}")
        
        # Auto-configure based on GPU
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            
            optimal = get_gpu_optimal_config(gpu_name)
            config.agent_config.batch_size = optimal["batch_size"]
            config.agent_config.buffer_size = optimal["buffer_size"]
            config.agent_config.hidden_dims = optimal["hidden_dims"]
            config.agent_config.num_quantiles = optimal["num_quantiles"]
            
            logger.info(f"Optimal settings: batch={optimal['batch_size']}, buffer={optimal['buffer_size']}")
        
        # Update agent device
        config.agent_config.device = str(self.device)
        
        # Create environment
        if env is not None:
            self.env = env
        else:
            self.env = TradingEnv(
                config=config.env_config,
                price_data=price_data,
                feature_data=feature_data,
                tft_model=tft_model,
            )
        
        # Create agent
        if agent is not None:
            self.agent = agent
        else:
            self.agent = QRDQNAgent(config.agent_config)
        
        # Reward function
        self.reward_fn = DifferentialSharpeRatio(config.reward_config)
        
        # Logging
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.global_step = 0
        self.episode = 0
        self.best_eval_return = -float('inf')
        
        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_returns: List[float] = []
    
    def train(self) -> Dict:
        """
        Run training loop.
        
        Returns:
            Training statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING RL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Total timesteps: {self.config.total_timesteps:,}")
        logger.info(f"Batch size: {self.config.agent_config.batch_size}")
        logger.info(f"Buffer size: {self.config.agent_config.buffer_size:,}")
        logger.info(f"Using PopArt: {self.config.agent_config.use_popart}")
        
        start_time = time.time()
        
        # Training loop
        state, _ = self.env.reset()
        self.reward_fn.reset()
        episode_reward = 0
        episode_length = 0
        last_action = None
        
        while self.global_step < self.config.total_timesteps:
            # Select action
            action = self.agent.select_action(state)
            
            # Take step
            next_state, raw_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Compute DSR reward
            step_return = info.get("step_return", raw_reward)
            action_changed = (last_action is not None and action != last_action)
            
            dsr_reward = self.reward_fn.compute(
                step_return=step_return,
                position=self.env.position,
                drawdown=info.get("drawdown", 0),
                action_changed=action_changed,
            )
            
            # Store transition
            self.agent.store_transition(state, action, dsr_reward, next_state, done)
            
            # Train agent
            train_metrics = self.agent.train_step()
            
            # Update state
            state = next_state
            last_action = action
            episode_reward += dsr_reward
            episode_length += 1
            self.global_step += 1
            
            # Episode done
            if done:
                self.episode += 1
                total_return = info.get("total_return", 0)
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_returns.append(total_return)
                
                # Log episode
                self.writer.add_scalar("episode/reward", episode_reward, self.episode)
                self.writer.add_scalar("episode/length", episode_length, self.episode)
                self.writer.add_scalar("episode/return", total_return, self.episode)
                self.writer.add_scalar("episode/sharpe", self.env.get_sharpe_ratio(), self.episode)
                
                # Reset
                state, _ = self.env.reset()
                self.reward_fn.reset()
                episode_reward = 0
                episode_length = 0
                last_action = None
            
            # Logging
            if self.global_step % self.config.log_frequency == 0:
                self._log_training(train_metrics)
            
            # Evaluation
            if self.global_step % self.config.eval_frequency == 0:
                eval_return = self._evaluate()
                
                if eval_return > self.best_eval_return:
                    self.best_eval_return = eval_return
                    self._save_checkpoint("best_agent.pt")
            
            # Checkpoint
            if self.global_step % self.config.save_frequency == 0:
                self._save_checkpoint(f"agent_step_{self.global_step}.pt")
        
        # Final save
        self._save_checkpoint("final_agent.pt")
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed/3600:.2f} hours")
        logger.info(f"Episodes: {self.episode}")
        logger.info(f"Best eval return: {self.best_eval_return:.4f}")
        
        self.writer.close()
        
        return {
            "total_steps": self.global_step,
            "episodes": self.episode,
            "best_eval_return": self.best_eval_return,
            "elapsed_time": elapsed,
        }
    
    def _evaluate(self) -> float:
        """
        Evaluate the agent.
        
        Returns:
            Mean episode return
        """
        logger.info(f"Evaluating at step {self.global_step}...")
        
        eval_returns = []
        eval_sharpes = []
        
        for _ in range(self.config.eval_episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                state, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            
            eval_returns.append(info.get("total_return", 0))
            eval_sharpes.append(self.env.get_sharpe_ratio())
        
        mean_return = np.mean(eval_returns)
        mean_sharpe = np.mean(eval_sharpes)
        
        self.writer.add_scalar("eval/mean_return", mean_return, self.global_step)
        self.writer.add_scalar("eval/mean_sharpe", mean_sharpe, self.global_step)
        self.writer.add_scalar("eval/std_return", np.std(eval_returns), self.global_step)
        
        logger.info(f"Eval: Return={mean_return:.4f} (+/- {np.std(eval_returns):.4f}), Sharpe={mean_sharpe:.2f}")
        
        return mean_return
    
    def _log_training(self, train_metrics: Dict):
        """Log training metrics."""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)
        
        # Agent stats
        agent_stats = self.agent.get_stats()
        self.writer.add_scalar("agent/epsilon", agent_stats["epsilon"], self.global_step)
        self.writer.add_scalar("agent/buffer_size", agent_stats["buffer_size"], self.global_step)
        
        if "popart_mean" in agent_stats:
            self.writer.add_scalar("popart/mean", agent_stats["popart_mean"], self.global_step)
            self.writer.add_scalar("popart/std", agent_stats["popart_std"], self.global_step)
        
        # DSR stats
        dsr_stats = self.reward_fn.get_stats()
        self.writer.add_scalar("dsr/A", dsr_stats["A"], self.global_step)
        self.writer.add_scalar("dsr/B", dsr_stats["B"], self.global_step)
        self.writer.add_scalar("dsr/sharpe_estimate", dsr_stats["sharpe_estimate"], self.global_step)
        
        # Log to console periodically
        if self.global_step % (self.config.log_frequency * 10) == 0:
            logger.info(
                f"Step {self.global_step:,} | "
                f"Loss: {train_metrics.get('loss', 0):.4f} | "
                f"Epsilon: {agent_stats['epsilon']:.3f} | "
                f"Buffer: {agent_stats['buffer_size']:,}"
            )
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        self.agent.save(str(path))
        logger.info(f"Saved checkpoint: {path}")


def train_rl_agent(
    tft_model: Optional[torch.nn.Module] = None,
    price_data: Optional[np.ndarray] = None,
    feature_data: Optional[np.ndarray] = None,
    total_timesteps: int = 500000,
    device: str = "auto",
    **kwargs,
) -> Tuple[QRDQNAgent, Dict]:
    """
    Convenience function to train an RL agent.
    
    Args:
        tft_model: Pretrained TFT model for context extraction
        price_data: Price series for the environment
        feature_data: Feature data for TFT (if using)
        total_timesteps: Total training steps
        device: Device to use
        **kwargs: Additional config options
    
    Returns:
        (trained_agent, training_stats)
    """
    config = RLTrainingConfig(
        total_timesteps=total_timesteps,
        device=device,
    )
    
    # Apply any kwargs to config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.agent_config, key):
            setattr(config.agent_config, key, value)
        elif hasattr(config.env_config, key):
            setattr(config.env_config, key, value)
    
    trainer = RLTrainer(
        config=config,
        tft_model=tft_model,
        price_data=price_data,
        feature_data=feature_data,
    )
    
    stats = trainer.train()
    
    return trainer.agent, stats
