"""
QR-DQN (Quantile Regression DQN) Agent.

From: "Distributional Reinforcement Learning with Quantile Regression" (Dabney et al., 2017)

QR-DQN models the full distribution of returns, not just expected value.
This is crucial for:
1. Risk-aware decision making
2. Better exploration through uncertainty
3. Robustness to non-stationary rewards (crypto volatility)

Combined with PopArt normalization for stable learning across reward scales.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.rl.popart import PopArtLayer, PopArtNormalizer

logger = logging.getLogger(__name__)


@dataclass
class QRDQNConfig:
    """Configuration for QR-DQN agent."""
    
    # Network architecture
    state_dim: int = 64 + 15  # TFT context + account state
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    num_actions: int = 3  # Hold, Buy, Sell
    num_quantiles: int = 32  # Number of quantiles to estimate
    
    # Training
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    
    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 10000  # Minimum samples before training
    
    # Target network
    target_update_freq: int = 1000  # Steps between target updates
    soft_update_tau: float = 0.005  # For soft updates (if used)
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50000
    
    # PopArt
    use_popart: bool = True
    popart_beta: float = 0.0001
    
    # Regularization
    dropout: float = 0.1
    gradient_clip: float = 10.0
    
    # Device
    device: str = "cuda"


class QRDQNNetwork(nn.Module):
    """
    Quantile Regression DQN Network.
    
    Outputs a distribution of Q-values for each action.
    """
    
    def __init__(self, config: QRDQNConfig):
        super().__init__()
        
        self.config = config
        self.num_quantiles = config.num_quantiles
        self.num_actions = config.num_actions
        
        # Build feature extractor
        layers = []
        in_dim = config.state_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = config.hidden_dims[-1]
        
        # Output layer (with or without PopArt)
        if config.use_popart:
            self.output_layer = PopArtLayer(
                input_dim=self.feature_dim,
                num_actions=config.num_actions,
                num_quantiles=config.num_quantiles,
                beta=config.popart_beta,
            )
        else:
            self.output_layer = nn.Linear(
                self.feature_dim,
                config.num_actions * config.num_quantiles,
            )
        
        # Quantile fractions (tau_i for i=1..N)
        # Fixed quantile midpoints for QR-DQN
        taus = torch.linspace(0, 1, config.num_quantiles + 1)
        self.register_buffer(
            "tau",
            (taus[:-1] + taus[1:]) / 2  # Midpoints
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            Quantile values (batch_size, num_actions, num_quantiles)
        """
        features = self.feature_extractor(state)
        
        if isinstance(self.output_layer, PopArtLayer):
            return self.output_layer(features)
        else:
            output = self.output_layer(features)
            return output.view(-1, self.num_actions, self.num_quantiles)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get expected Q-values (mean of quantiles).
        
        Args:
            state: State tensor
        
        Returns:
            Q-values (batch_size, num_actions)
        """
        quantiles = self.forward(state)
        
        if isinstance(self.output_layer, PopArtLayer):
            # Denormalize quantiles and take mean
            features = self.feature_extractor(state)
            raw_quantiles = self.output_layer(features)
            denorm_quantiles = self.output_layer.popart.denormalize(
                raw_quantiles.view(-1, self.num_actions * self.num_quantiles)
            ).view(-1, self.num_actions, self.num_quantiles)
            return denorm_quantiles.mean(dim=-1)
        else:
            return quantiles.mean(dim=-1)
    
    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor (1, state_dim)
            epsilon: Exploration probability
        
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        
        with torch.no_grad():
            q_values = self.get_q_values(state)
            return q_values.argmax(dim=-1).item()
    
    def get_action_with_uncertainty(
        self,
        state: torch.Tensor,
    ) -> Tuple[int, float]:
        """
        Select action and return uncertainty estimate.
        
        Uncertainty = std of quantiles for best action.
        
        Returns:
            (action, uncertainty)
        """
        with torch.no_grad():
            quantiles = self.forward(state)  # (1, num_actions, num_quantiles)
            q_values = quantiles.mean(dim=-1)  # (1, num_actions)
            
            best_action = q_values.argmax(dim=-1).item()
            best_quantiles = quantiles[0, best_action]
            uncertainty = best_quantiles.std().item()
            
            return best_action, uncertainty


class ReplayBuffer:
    """
    Experience replay buffer for RL training.
    """
    
    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        idx = self.position
        
        self.states[idx] = torch.from_numpy(state)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = torch.from_numpy(next_state)
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device),
        )
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples experiences with probability proportional to TD error.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001,
        device: str = "cpu",
    ):
        super().__init__(capacity, state_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        self.priorities = np.zeros(capacity, dtype=np.float32)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience with max priority."""
        self.priorities[self.position] = self.max_priority
        super().push(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample with prioritization."""
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device),
            weights,
            indices,
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class QRDQNAgent:
    """
    QR-DQN Agent with PopArt normalization.
    
    Combines:
    1. Distributional RL (QR-DQN) for risk-aware decisions
    2. PopArt for stable learning across reward scales
    3. Prioritized replay for sample efficiency
    """
    
    def __init__(self, config: QRDQNConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = QRDQNNetwork(config).to(self.device)
        self.target_net = QRDQNNetwork(config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = AdamW(
            self.online_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=100000,
            eta_min=config.learning_rate * 0.1,
        )
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            state_dim=config.state_dim,
            device=self.device,
        )
        
        # Training state
        self.train_steps = 0
        self.epsilon = config.epsilon_start
        
        # Quantile huber loss threshold
        self.kappa = 1.0  # Huber loss threshold
        
        logger.info(f"QR-DQN Agent initialized on {self.device}")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.online_net.parameters()):,}")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online_net.get_action(state_tensor, self.epsilon)
    
    def select_action_with_uncertainty(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action and return uncertainty."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online_net.get_action_with_uncertainty(state_tensor)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return {"loss": 0.0}
        
        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)
        
        if len(batch) == 7:
            # Prioritized replay
            states, actions, rewards, next_states, dones, weights, indices = batch
        else:
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones_like(rewards)
            indices = None
        
        # Get current quantiles
        current_quantiles = self.online_net(states)  # (batch, actions, quantiles)
        
        # Get quantiles for taken actions
        actions_expanded = actions.unsqueeze(-1).expand(-1, 1, self.config.num_quantiles)
        current_q = current_quantiles.gather(1, actions_expanded).squeeze(1)  # (batch, quantiles)
        
        # Compute target quantiles
        with torch.no_grad():
            # Double DQN: use online net to select action, target net for values
            next_q_online = self.online_net.get_q_values(next_states)
            next_actions = next_q_online.argmax(dim=-1, keepdim=True)
            
            next_quantiles = self.target_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, 1, self.config.num_quantiles)
            next_q = next_quantiles.gather(1, next_actions_expanded).squeeze(1)
            
            # TD target
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # Update PopArt if enabled
        if self.config.use_popart and isinstance(self.online_net.output_layer, PopArtLayer):
            # Use running mean/std for target normalization
            # PopArt stats are tracked but we normalize per-quantile
            target_mean = target_q.mean()
            target_std = target_q.std() + 1e-8
            target_q_norm = (target_q - target_mean) / target_std
            
            # Update PopArt stats for output layer preservation
            # Flatten targets for stats update
            batch_size = target_q.shape[0]
            flat_targets = target_q.mean(dim=-1, keepdim=True).expand(-1, self.config.num_actions * self.config.num_quantiles)
            self.online_net.output_layer.popart.update_stats(flat_targets)
        else:
            target_q_norm = target_q
        
        # Quantile Huber loss
        loss, td_errors = self._compute_quantile_huber_loss(current_q, target_q_norm, weights)
        
        # Update priorities
        if indices is not None:
            self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            self.config.gradient_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        self._decay_epsilon()
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q.mean().item(),
            "q_std": current_q.std().item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    def _compute_quantile_huber_loss(
        self,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantile Huber loss.
        
        Args:
            current_q: Current quantile estimates (batch, num_quantiles)
            target_q: Target quantile values (batch, num_quantiles)
            weights: Importance sampling weights
        
        Returns:
            (loss, td_errors)
        """
        batch_size = current_q.shape[0]
        num_quantiles = self.config.num_quantiles
        
        # Compute pairwise TD errors
        # target_q: (batch, num_quantiles) -> (batch, 1, num_quantiles)
        # current_q: (batch, num_quantiles) -> (batch, num_quantiles, 1)
        td_error = target_q.unsqueeze(1) - current_q.unsqueeze(2)  # (batch, N, N)
        
        # Huber loss
        huber_loss = torch.where(
            td_error.abs() <= self.kappa,
            0.5 * td_error.pow(2),
            self.kappa * (td_error.abs() - 0.5 * self.kappa),
        )
        
        # Quantile weights
        tau = self.online_net.tau.view(1, -1, 1)  # (1, N, 1)
        quantile_weights = torch.abs(tau - (td_error.detach() < 0).float())
        
        # Quantile Huber loss
        quantile_loss = (quantile_weights * huber_loss).sum(dim=-1).mean(dim=-1)
        
        # Apply importance sampling weights
        loss = (weights.squeeze() * quantile_loss).mean()
        
        # TD errors for priority update (mean absolute error)
        td_errors = td_error.abs().mean(dim=(1, 2))
        
        return loss, td_errors
    
    def _update_target_network(self):
        """Hard update target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Updated target network")
    
    def _soft_update_target_network(self):
        """Soft update target network."""
        tau = self.config.soft_update_tau
        for target_param, online_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )
    
    def _decay_epsilon(self):
        """Decay exploration epsilon."""
        decay_rate = (
            (self.config.epsilon_start - self.config.epsilon_end) /
            self.config.epsilon_decay_steps
        )
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon - decay_rate
        )
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
            "config": self.config,
        }, path)
        logger.info(f"Saved agent to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]
        
        logger.info(f"Loaded agent from {path}")
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        stats = {
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
        
        # Add PopArt stats if enabled
        if self.config.use_popart and isinstance(self.online_net.output_layer, PopArtLayer):
            popart_stats = self.online_net.output_layer.popart.get_stats()
            stats["popart_mean"] = popart_stats["mean"].mean()
            stats["popart_std"] = popart_stats["std"].mean()
        
        return stats
