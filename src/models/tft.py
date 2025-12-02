"""
Temporal Fusion Transformer (TFT) Implementation.

The TFT serves as the "Eyes" of our trading system - it understands
the market state and produces a rich context vector for the RL agent.

Key Features:
1. Variable Selection Networks - Filter out irrelevant features dynamically
2. Gated Residual Networks - Control information flow
3. Multi-Head Attention - Capture long-range temporal dependencies
4. Quantile Outputs - Predict distribution, not just point estimates

Architecture Overview:
    Input Features → Variable Selection → LSTM Encoder → 
    Self-Attention → GRN → Quantile Outputs

References:
- Lim et al. (2021): "Temporal Fusion Transformers for Interpretable 
  Multi-horizon Time Series Forecasting"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .components import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    PositionalEncoding,
    TemporalBlock,
    QuantileLoss,
)


@dataclass
class TFTConfig:
    """Configuration for the Temporal Fusion Transformer."""
    
    # Feature dimensions
    num_time_varying_features: int = 12  # FracDiff price, OFI, etc.
    num_static_features: int = 0  # Symbol embeddings, etc.
    
    # Feature names for interpretability
    time_varying_feature_names: List[str] = field(default_factory=lambda: [
        "price_ffd", "volume_ffd", "ofi_level_0", "ofi_level_5",
        "ofi_1s", "ofi_5s", "ofi_zscore_1s", "spread",
        "trade_imbalance", "fdi", "volatility", "global_risk_score",
    ])
    
    # Model dimensions
    hidden_dim: int = 64
    embed_dim: int = 64
    lstm_layers: int = 1
    
    # Attention
    num_heads: int = 4
    num_attention_layers: int = 2
    
    # Sequence lengths
    encoder_length: int = 100  # Look back 100 timesteps
    decoder_length: int = 10  # Predict 10 steps ahead
    
    # Output
    num_quantiles: int = 3  # 10%, 50%, 90%
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    output_dim: int = 1  # Predicting volatility or returns
    
    # Regularization
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration."""
        assert len(self.quantiles) == self.num_quantiles
        assert self.embed_dim % self.num_heads == 0


class InputEmbedding(nn.Module):
    """
    Embed raw input features into a unified representation.
    
    Each feature type gets its own linear projection to embed_dim,
    ensuring they all live in the same representational space.
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Project each time-varying feature to embed_dim
        self.time_varying_embeddings = nn.ModuleList([
            nn.Linear(1, config.embed_dim)
            for _ in range(config.num_time_varying_features)
        ])
        
        # Optional static embeddings
        if config.num_static_features > 0:
            self.static_embeddings = nn.ModuleList([
                nn.Linear(1, config.embed_dim)
                for _ in range(config.num_static_features)
            ])
        else:
            self.static_embeddings = None
    
    def forward(
        self,
        time_varying: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Embed input features.
        
        Args:
            time_varying: (batch, seq, num_time_varying_features)
            static: Optional (batch, num_static_features)
        
        Returns:
            Tuple of:
            - List of embedded time-varying features, each (batch, seq, embed_dim)
            - Optional list of embedded static features
        """
        # Embed each time-varying feature
        time_varying_embedded = []
        for i, embed in enumerate(self.time_varying_embeddings):
            feature = time_varying[..., i:i+1]  # (batch, seq, 1)
            time_varying_embedded.append(embed(feature))
        
        # Embed static features
        static_embedded = None
        if static is not None and self.static_embeddings is not None:
            static_embedded = []
            for i, embed in enumerate(self.static_embeddings):
                feature = static[..., i:i+1]  # (batch, 1)
                static_embedded.append(embed(feature))
        
        return time_varying_embedded, static_embedded


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for market state understanding.
    
    This model learns to:
    1. Select relevant features for the current market regime
    2. Encode temporal patterns via LSTM + self-attention
    3. Predict quantiles of future volatility/returns
    
    The encoder output (context vector) can be frozen and used
    as input to the RL agent.
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = InputEmbedding(config)
        
        # Variable selection for encoder inputs
        self.encoder_variable_selection = VariableSelectionNetwork(
            num_features=config.num_time_varying_features,
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embed_dim,
            dropout=config.dropout,
        )
        
        # LSTM encoder for temporal processing
        self.lstm_encoder = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
        )
        
        # Post-LSTM gating
        self.lstm_grn = GatedResidualNetwork(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embed_dim,
            dropout=config.dropout,
        )
        
        # Positional encoding for attention
        self.positional_encoding = PositionalEncoding(
            embed_dim=config.embed_dim,
            max_len=config.encoder_length + config.decoder_length,
            dropout=config.dropout,
        )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            TemporalBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_attention_layers)
        ])
        
        # Output projection for each prediction horizon
        self.output_grn = GatedResidualNetwork(
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embed_dim,
            dropout=config.dropout,
        )
        
        # Quantile output heads (one per horizon)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(config.embed_dim, config.num_quantiles * config.output_dim)
            for _ in range(config.decoder_length)
        ])
        
        # Loss function
        self.quantile_loss = QuantileLoss(quantiles=config.quantiles)
        
        # Context vector projection (for RL agent)
        self.context_projection = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Store attention weights for interpretability
        self.attention_weights = None
        self.variable_weights = None
    
    def forward(
        self,
        time_varying: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        return_context: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            time_varying: (batch, encoder_length, num_time_varying_features)
            static: Optional (batch, num_static_features)
            return_context: Whether to return the context vector
        
        Returns:
            If return_context=False:
                predictions: (batch, decoder_length, num_quantiles)
            If return_context=True:
                Tuple of (predictions, context_vector)
        """
        batch_size = time_varying.size(0)
        
        # Embed inputs
        time_varying_embedded, static_embedded = self.input_embedding(
            time_varying, static
        )
        
        # Variable selection
        selected_features, var_weights = self.encoder_variable_selection(
            time_varying_embedded
        )
        self.variable_weights = var_weights  # Store for interpretability
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm_encoder(selected_features)
        
        # Post-LSTM gating
        lstm_out = self.lstm_grn(lstm_out)
        
        # Add positional encoding
        lstm_out = self.positional_encoding(lstm_out)
        
        # Self-attention layers
        attention_out = lstm_out
        all_attention_weights = []
        
        for attention_layer in self.attention_layers:
            attention_out, attn_weights = attention_layer(attention_out)
            all_attention_weights.append(attn_weights)
        
        self.attention_weights = all_attention_weights  # Store for interpretability
        
        # Output processing
        output = self.output_grn(attention_out)
        
        # Context vector: use the last timestep's hidden state
        context = self.context_projection(output[:, -1, :])  # (batch, embed_dim)
        
        # Generate quantile predictions for each horizon
        predictions = []
        for t, quantile_head in enumerate(self.quantile_heads):
            # Use the (encoder_length - decoder_length + t) position
            # This gives us a sliding window prediction
            idx = min(t, output.size(1) - 1)
            pred = quantile_head(output[:, idx, :])  # (batch, num_quantiles)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # (batch, decoder_length, num_quantiles)
        
        if return_context:
            return predictions, context
        return predictions
    
    def compute_loss(
        self,
        time_varying: torch.Tensor,
        targets: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            time_varying: (batch, encoder_length, num_features)
            targets: (batch, decoder_length) - future values to predict
            static: Optional static features
        
        Returns:
            Dictionary with loss components
        """
        predictions = self.forward(time_varying, static, return_context=False)
        
        # Quantile loss
        loss = self.quantile_loss(predictions, targets)
        
        return {
            "loss": loss,
            "quantile_loss": loss,
        }
    
    def get_context_vector(
        self,
        time_varying: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the context vector for the RL agent.
        
        This is the main interface for transferring learned representations
        to the downstream RL agent.
        
        Args:
            time_varying: (batch, encoder_length, num_features)
            static: Optional static features
        
        Returns:
            Context vector (batch, embed_dim)
        """
        _, context = self.forward(time_varying, static, return_context=True)
        return context
    
    def get_interpretability_info(self) -> Dict:
        """
        Get interpretability information.
        
        Returns variable selection weights and attention patterns
        for analysis.
        """
        return {
            "variable_weights": self.variable_weights,
            "attention_weights": self.attention_weights,
            "feature_names": self.config.time_varying_feature_names,
        }
    
    def freeze_encoder(self):
        """
        Freeze the encoder for transfer learning to RL.
        
        Only the context projection layer remains trainable.
        """
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze context projection
        for param in self.context_projection.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class TFTForecaster:
    """
    High-level wrapper for TFT training and inference.
    
    Handles:
    - Device management (CPU/GPU)
    - Training loop with mixed precision
    - Checkpointing
    - Inference
    """
    
    def __init__(
        self,
        config: TFTConfig,
        checkpoint_path: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = TemporalFusionTransformer(config)
        self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def train_step(
        self,
        time_varying: torch.Tensor,
        targets: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step with mixed precision.
        
        Args:
            time_varying: Input features (batch, seq, features)
            targets: Target values (batch, decoder_length)
            static: Optional static features
        
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        time_varying = time_varying.to(self.device)
        targets = targets.to(self.device)
        if static is not None:
            static = static.to(self.device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
            losses = self.model.compute_loss(time_varying, targets, static)
        
        # Backward pass
        self.scaler.scale(losses["loss"]).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def predict(
        self,
        time_varying: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            time_varying: (batch, encoder_length, num_features)
            static: Optional static features
        
        Returns:
            Predictions (batch, decoder_length, num_quantiles)
        """
        self.model.eval()
        
        time_varying = time_varying.to(self.device)
        if static is not None:
            static = static.to(self.device)
        
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
            predictions = self.model(time_varying, static, return_context=False)
        
        return predictions.float()
    
    @torch.no_grad()
    def get_context(
        self,
        time_varying: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get context vector for RL agent.
        
        Args:
            time_varying: (batch, encoder_length, num_features)
            static: Optional static features
        
        Returns:
            Context vector (batch, embed_dim)
        """
        self.model.eval()
        
        time_varying = time_varying.to(self.device)
        if static is not None:
            static = static.to(self.device)
        
        return self.model.get_context_vector(time_varying, static)
    
    def save_checkpoint(self, path: str, epoch: int = 0, best_loss: float = float('inf')):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "best_loss": best_loss,
        }, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        return checkpoint
