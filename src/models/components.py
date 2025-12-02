"""
TFT Building Blocks - Gated Residual Networks, Variable Selection, and Attention.

These components form the core of the Temporal Fusion Transformer architecture.

References:
- Lim et al. (2021): "Temporal Fusion Transformers for Interpretable 
  Multi-horizon Time Series Forecasting"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


class GLU(nn.Module):
    """
    Gated Linear Unit activation.
    
    GLU(x) = σ(Wx + b) ⊙ (Vx + c)
    
    This allows the network to control information flow,
    effectively "turning off" irrelevant features.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x[..., :self.output_dim] * torch.sigmoid(x[..., self.output_dim:])


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    
    Features:
    - Skip connections for stable gradient flow
    - Gating mechanism to filter irrelevant information
    - Optional context vector for conditional processing
    
    Architecture:
        η = ELU(W_1 * x + W_2 * c + b_1)  # c is optional context
        η = W_3 * η + b_2
        GLU(η) + x  # gated skip connection
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Layer 1: Input projection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Optional context projection
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
        
        # Layer 2: Hidden to output
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Gated skip connection
        self.gate = GLU(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Skip projection if dimensions don't match
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
        
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, ..., input_dim)
            context: Optional context vector (batch, ..., context_dim)
        
        Returns:
            Output tensor (batch, ..., output_dim)
        """
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x
        
        # Layer 1 with optional context
        hidden = self.fc1(x)
        if context is not None and self.context_fc is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        
        # Layer 2
        hidden = self.dropout(self.fc2(hidden))
        
        # Gated skip connection
        output = self.layer_norm(self.gate(hidden) + skip)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - Dynamically selects relevant features.
    
    This is crucial for handling noisy financial data where not all
    indicators are relevant at all times (e.g., RSI during strong trends).
    
    The network learns to weight each input feature based on:
    1. The feature's intrinsic importance (static)
    2. The current context (dynamic)
    
    Output: Weighted combination of transformed features
    """
    
    def __init__(
        self,
        num_features: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Transform each feature independently
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            for _ in range(num_features)
        ])
        
        # Selection weights GRN
        self.selection_grn = GatedResidualNetwork(
            input_dim=num_features * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        features: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: List of feature tensors, each (batch, seq, input_dim)
            context: Optional context (batch, seq, context_dim)
        
        Returns:
            Tuple of:
            - Combined features (batch, seq, output_dim)
            - Selection weights (batch, seq, num_features)
        """
        # Transform each feature
        transformed = []
        for i, (feature, grn) in enumerate(zip(features, self.feature_grns)):
            transformed.append(grn(feature))
        
        # Stack for weighted combination: (batch, seq, num_features, output_dim)
        transformed_stack = torch.stack(transformed, dim=-2)
        
        # Compute selection weights
        flattened = torch.cat(features, dim=-1)  # (batch, seq, num_features * input_dim)
        weights = self.selection_grn(flattened, context)  # (batch, seq, num_features)
        weights = self.softmax(weights)
        
        # Weighted combination
        # weights: (batch, seq, num_features, 1)
        # transformed_stack: (batch, seq, num_features, output_dim)
        weights_expanded = weights.unsqueeze(-1)
        combined = (transformed_stack * weights_expanded).sum(dim=-2)
        
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for temporal patterns.
    
    Key differences from standard attention:
    1. Shares values across heads (interpretability)
    2. Adds learned position-aware attention biases
    3. Returns attention weights for analysis
    
    This allows the model to look back 100-500+ timesteps
    and find relevant historical patterns.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K projections per head
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        
        # Shared V projection (interpretability)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal masking for autoregressive modeling.
        
        Args:
            query: (batch, seq_q, embed_dim)
            key: (batch, seq_k, embed_dim)
            value: (batch, seq_v, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Tuple of:
            - Output (batch, seq_q, embed_dim)
            - Attention weights (batch, num_heads, seq_q, seq_k)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # (batch, seq, embed) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask (causal or padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Causal mask for autoregressive
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position awareness.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, embed_dim)
        
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class QuantileLoss(nn.Module):
    """
    Quantile Loss for probabilistic forecasting.
    
    Instead of predicting a single value, predict multiple quantiles
    to understand the distribution of possible outcomes.
    
    This is crucial for risk management - we want to know not just
    the expected price, but also the tail risks (5th, 95th percentiles).
    
    Loss = Σ_q Σ_t max(q(y-ŷ), (q-1)(y-ŷ))
    """
    
    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: (batch, seq, num_quantiles) or (batch, num_quantiles)
            targets: (batch, seq) or (batch,)
        
        Returns:
            Scalar loss
        """
        if targets.dim() < predictions.dim():
            targets = targets.unsqueeze(-1)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[..., i:i+1]
            error = targets - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        
        total_loss = torch.cat(losses, dim=-1).mean()
        return total_loss


class TemporalBlock(nn.Module):
    """
    Single temporal processing block combining attention and GRN.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = InterpretableMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.attention_grn = GatedResidualNetwork(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.layer_norm(x + attn_out)
        
        # GRN with residual
        x = self.attention_grn(x) + x
        
        return x, attn_weights
