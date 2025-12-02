"""
PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets) Normalization.

From: "Learning values across many orders of magnitude" (DeepMind, 2016)

PopArt solves the problem of:
1. Reward scales changing during training (crypto pumps/crashes)
2. Non-stationary reward distributions
3. Maintaining stable learning despite extreme values

It works by:
1. Tracking running mean and std of targets
2. Normalizing targets before computing loss
3. De-normalizing outputs for action selection
4. Preserving the output mapping when stats change
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PopArtNormalizer(nn.Module):
    """
    PopArt normalization layer.
    
    Wraps a linear layer to provide adaptive normalization.
    When the running statistics change, the weights and biases
    are adjusted to preserve the output mapping.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        beta: float = 0.0001,  # Update rate for running stats
        epsilon: float = 1e-4,  # Minimum std for stability
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta = beta
        self.epsilon = epsilon
        
        # Output layer (will be rescaled by PopArt)
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Running statistics (per output)
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("mean_sq", torch.ones(output_dim))  # E[x^2]
        self.register_buffer("std", torch.ones(output_dim))
        self.register_buffer("count", torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - outputs are in normalized space.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Normalized output (batch_size, output_dim)
        """
        return self.linear(x)
    
    def denormalize(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized output back to original scale.
        
        Args:
            normalized_output: Output from forward() in normalized space
        
        Returns:
            Output in original scale
        """
        return normalized_output * self.std + self.mean
    
    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        """
        Normalize targets for loss computation.
        
        Args:
            target: Raw target values
        
        Returns:
            Normalized targets
        """
        return (target - self.mean) / self.std
    
    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        """
        Update running statistics and adjust weights to preserve outputs.
        
        This is the key PopArt operation:
        1. Compute new running mean and std
        2. Adjust linear layer weights/biases so outputs don't change
        
        Args:
            targets: Batch of target values (batch_size,) or (batch_size, output_dim)
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).expand(-1, self.output_dim)
        
        # Store old stats
        old_mean = self.mean.clone()
        old_std = self.std.clone()
        
        # Update running statistics using Welford's algorithm variant
        batch_size = targets.shape[0]
        batch_mean = targets.mean(dim=0)
        batch_mean_sq = (targets ** 2).mean(dim=0)
        
        # Exponential moving average update
        new_count = self.count + batch_size
        
        if self.count == 0:
            # First update
            self.mean.copy_(batch_mean)
            self.mean_sq.copy_(batch_mean_sq)
        else:
            # EMA update
            self.mean.copy_(self.mean * (1 - self.beta) + batch_mean * self.beta)
            self.mean_sq.copy_(self.mean_sq * (1 - self.beta) + batch_mean_sq * self.beta)
        
        self.count.copy_(new_count)
        
        # Compute new std
        variance = self.mean_sq - self.mean ** 2
        self.std.copy_(torch.sqrt(torch.clamp(variance, min=self.epsilon ** 2)))
        
        # Preserve outputs: adjust weights and biases
        # If old output was: y = Wx + b
        # And old normalized: y_norm = (y - old_mean) / old_std
        # Then new output should give same y_norm with new stats
        # So: W_new = W * (old_std / new_std)
        #     b_new = (old_std * b + old_mean - new_mean) / new_std
        
        std_ratio = old_std / self.std
        
        # Adjust weights (scale each output dimension)
        self.linear.weight.data *= std_ratio.unsqueeze(-1)
        
        # Adjust biases
        self.linear.bias.data = (
            old_std * self.linear.bias.data + old_mean - self.mean
        ) / self.std
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "mean": self.mean.cpu().numpy(),
            "std": self.std.cpu().numpy(),
            "count": self.count.item(),
        }


class PopArtLayer(nn.Module):
    """
    PopArt-normalized output layer for Q-networks.
    
    Specifically designed for QR-DQN where we need:
    1. Multiple quantile outputs
    2. Stable learning across reward scales
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        num_quantiles: int = 32,
        beta: float = 0.0001,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        # Separate PopArt normalizer for each action
        # Output dim = num_quantiles for each action
        self.popart = PopArtNormalizer(
            input_dim=input_dim,
            output_dim=num_actions * num_quantiles,
            beta=beta,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Q-value quantiles (batch_size, num_actions, num_quantiles)
        """
        output = self.popart(x)  # (batch, num_actions * num_quantiles)
        return output.view(-1, self.num_actions, self.num_quantiles)
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get expected Q-values (mean of quantiles).
        
        Args:
            x: Input features
        
        Returns:
            Q-values (batch_size, num_actions)
        """
        quantiles = self.forward(x)
        # Denormalize and take mean
        denorm_quantiles = self.popart.denormalize(
            quantiles.view(-1, self.num_actions * self.num_quantiles)
        ).view(-1, self.num_actions, self.num_quantiles)
        
        return denorm_quantiles.mean(dim=-1)
    
    @torch.no_grad()
    def update_stats(self, td_targets: torch.Tensor):
        """
        Update PopArt statistics with TD targets.
        
        Args:
            td_targets: TD target values (batch_size, num_quantiles) 
                       or (batch_size,) for expected value
        """
        if td_targets.dim() == 1:
            # Expand for all quantiles
            td_targets = td_targets.unsqueeze(-1).expand(-1, self.num_quantiles)
        
        # Flatten for PopArt update
        batch_size = td_targets.shape[0]
        expanded = td_targets.unsqueeze(1).expand(-1, self.num_actions, -1)
        flat_targets = expanded.reshape(batch_size, -1)
        
        self.popart.update_stats(flat_targets)


class RunningMeanStd:
    """
    Simple running mean and std tracker (non-PopArt, for comparison).
    
    Uses Welford's online algorithm.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon  # Avoid div by zero
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """Update running stats with batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ):
        """Update stats from batch moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize values."""
        return x * np.sqrt(self.var + self.epsilon) + self.mean
