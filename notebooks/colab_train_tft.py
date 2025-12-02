"""
Google Colab Training Script for TFT Model.

Copy this entire file to a Colab notebook and run!
Supports: T4 (free tier), A100, H100

Usage in Colab:
    1. Upload this file or clone the repo
    2. Run: !python colab_train_tft.py
"""

# ============================================================================
# SETUP - Run this cell first
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages."""
    packages = [
        "torch",
        "numpy",
        "pandas", 
        "scipy",
        "statsmodels",
        "tensorboard",
        "einops",
    ]
    
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Uncomment to install packages
# install_packages()

# ============================================================================
# IMPORTS
# ============================================================================

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# GPU DETECTION AND OPTIMIZATION
# ============================================================================

def get_gpu_info():
    """Get GPU information and recommended settings."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "memory_gb": 0, "settings": {"batch_size": 16}}
    
    props = torch.cuda.get_device_properties(0)
    name = props.name
    memory_gb = props.total_memory / 1e9
    
    # Detect GPU type and set optimal parameters
    name_lower = name.lower()
    
    if "t4" in name_lower:
        settings = {
            "batch_size": 32,
            "gradient_accumulation": 2,
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "mixed_precision": "fp16",
        }
    elif "a100" in name_lower:
        settings = {
            "batch_size": 128,
            "gradient_accumulation": 1,
            "embed_dim": 128,
            "hidden_dim": 128,
            "num_layers": 4,
            "mixed_precision": "fp16",
        }
    elif "h100" in name_lower:
        settings = {
            "batch_size": 256,
            "gradient_accumulation": 1,
            "embed_dim": 256,
            "hidden_dim": 256,
            "num_layers": 6,
            "mixed_precision": "bf16",
        }
    elif "v100" in name_lower:
        settings = {
            "batch_size": 64,
            "gradient_accumulation": 1,
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "mixed_precision": "fp16",
        }
    else:
        # Default for unknown GPUs
        settings = {
            "batch_size": 32,
            "gradient_accumulation": 2,
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "mixed_precision": "fp16",
        }
    
    return {
        "name": name,
        "memory_gb": memory_gb,
        "settings": settings,
    }

# ============================================================================
# FRACTIONAL DIFFERENTIATION (Simplified for Colab)
# ============================================================================

def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute FFD weights."""
    weights = [1.0]
    k = 1
    while k < 10000:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])

def frac_diff_ffd(series: pd.Series, d: float = 0.4) -> np.ndarray:
    """Apply fractional differentiation."""
    weights = get_weights_ffd(d)
    result = np.convolve(series.values, weights, mode='valid')
    return result

# ============================================================================
# TFT COMPONENTS
# ============================================================================

class GLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.fc(x)
        return x[..., :self.output_dim] * torch.sigmoid(x[..., self.output_dim:])


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 context_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = GLU(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
    
    def forward(self, x, context=None):
        skip = self.skip_proj(x) if self.skip_proj else x
        hidden = self.fc1(x)
        if context is not None and self.context_fc:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        return self.layer_norm(self.gate(hidden) + skip)


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        B, L, _ = query.shape
        
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=query.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out), attn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions, targets):
        if targets.dim() < predictions.dim():
            targets = targets.unsqueeze(-1)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[..., i:i+1]
            error = targets - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        
        return torch.cat(losses, dim=-1).mean()


# ============================================================================
# TFT MODEL
# ============================================================================

@dataclass
class TFTConfig:
    num_features: int = 12
    embed_dim: int = 64
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    encoder_length: int = 100
    decoder_length: int = 10
    num_quantiles: int = 3
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    dropout: float = 0.1


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.num_features, config.embed_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            config.embed_dim, config.hidden_dim,
            batch_first=True, dropout=config.dropout
        )
        
        # Post-LSTM GRN
        self.lstm_grn = GatedResidualNetwork(
            config.hidden_dim, config.hidden_dim, config.embed_dim,
            dropout=config.dropout
        )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([
                InterpretableMultiHeadAttention(config.embed_dim, config.num_heads, config.dropout),
                GatedResidualNetwork(config.embed_dim, config.hidden_dim, config.embed_dim, dropout=config.dropout),
                nn.LayerNorm(config.embed_dim),
            ])
            for _ in range(config.num_layers)
        ])
        
        # Output heads
        self.output_grn = GatedResidualNetwork(
            config.embed_dim, config.hidden_dim, config.embed_dim,
            dropout=config.dropout
        )
        
        self.quantile_heads = nn.ModuleList([
            nn.Linear(config.embed_dim, config.num_quantiles)
            for _ in range(config.decoder_length)
        ])
        
        self.loss_fn = QuantileLoss(config.quantiles)
        
        # Context projection for RL
        self.context_proj = nn.Linear(config.embed_dim, config.embed_dim)
    
    def forward(self, x, return_context=False):
        # Input projection
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = self.lstm_grn(lstm_out)
        
        # Attention layers
        for attn, grn, norm in self.attention_layers:
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
            x = grn(x) + x
        
        # Output
        x = self.output_grn(x)
        
        # Context vector (last timestep)
        context = self.context_proj(x[:, -1, :])
        
        # Quantile predictions
        preds = torch.stack([head(x[:, i, :]) for i, head in enumerate(self.quantile_heads)], dim=1)
        
        if return_context:
            return preds, context
        return preds
    
    def compute_loss(self, x, targets):
        preds = self.forward(x)
        return {"loss": self.loss_fn(preds, targets)}


# ============================================================================
# DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, encoder_length=100, decoder_length=10):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        
        total_length = encoder_length + decoder_length
        self.indices = list(range(len(features) - total_length + 1))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.features[start:start + self.encoder_length]
        y = self.targets[start + self.encoder_length:start + self.encoder_length + self.decoder_length]
        return x, y


# ============================================================================
# TRAINING
# ============================================================================

def generate_synthetic_data(n_samples=50000):
    """
    Generate synthetic training data FOR TESTING ONLY.
    
    In production, replace this with real market data:
        1. Upload Parquet files from Phase 1 data ingestion
        2. Use MarketDataLoader from src/data/loader.py
        3. Or connect to live exchange WebSockets
    
    Example with real data:
        from src.data import MarketDataLoader, DataConfig
        config = DataConfig(data_dir="data/raw")
        loader = MarketDataLoader(config)
        train_ds, val_ds, test_ds = loader.prepare_datasets()
    """
    np.random.seed(42)
    
    # Volatility clustering
    volatility = np.zeros(n_samples)
    volatility[0] = 0.001
    for i in range(1, n_samples):
        volatility[i] = 0.00001 + 0.1 * volatility[i-1] + 0.85 * (np.random.randn() ** 2) * 0.0001
    
    # Price with FracDiff
    returns = np.random.randn(n_samples) * np.sqrt(volatility) + 0.00001
    log_prices = np.cumsum(returns) + np.log(50000)
    price_ffd = frac_diff_ffd(pd.Series(log_prices), d=0.4)
    
    # Pad
    pad = n_samples - len(price_ffd)
    price_ffd = np.concatenate([np.zeros(pad), price_ffd])
    
    # Features
    features = np.column_stack([
        price_ffd,
        np.random.randn(n_samples) * 0.1,  # volume_ffd
        np.random.randn(n_samples) * 10,   # ofi_level_0
        np.random.randn(n_samples) * 50,   # ofi_level_5
        np.cumsum(np.random.randn(n_samples) * 0.1),  # ofi_1s
        np.cumsum(np.random.randn(n_samples) * 0.05), # ofi_5s
        np.random.randn(n_samples),         # ofi_zscore_1s
        np.abs(np.random.randn(n_samples) * 0.1) + 0.01,  # spread
        np.random.randn(n_samples) * 100,   # trade_imbalance
        1.5 + np.random.randn(n_samples) * 0.2,  # fdi
        volatility,                          # volatility
        np.clip(0.5 + np.random.randn(n_samples) * 0.1, 0, 1),  # global_risk
    ])
    
    # Normalize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Target: future volatility
    targets = np.roll(volatility, -10)
    targets[-10:] = volatility[-10:]
    
    return features, targets


def train(config: TFTConfig, epochs=100, batch_size=32, lr=1e-3, device="cuda"):
    """Train the TFT model."""
    
    # Generate data
    logger.info("Generating synthetic data...")
    features, targets = generate_synthetic_data(50000)
    
    # Split
    split = int(len(features) * 0.8)
    train_ds = TimeSeriesDataset(features[:split], targets[:split], 
                                  config.encoder_length, config.decoder_length)
    val_ds = TimeSeriesDataset(features[split:], targets[split:],
                                config.encoder_length, config.decoder_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Model
    model = TemporalFusionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                loss = model.compute_loss(x, y)["loss"]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    loss = model.compute_loss(x, y)["loss"]
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_tft_model.pt")
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
    
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Get GPU info
    gpu_info = get_gpu_info()
    logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
    logger.info(f"Optimal settings: {gpu_info['settings']}")
    
    settings = gpu_info["settings"]
    
    # Create config with optimal settings
    config = TFTConfig(
        num_features=12,
        embed_dim=settings["embed_dim"],
        hidden_dim=settings["hidden_dim"],
        num_heads=4,
        num_layers=settings["num_layers"],
        encoder_length=100,
        decoder_length=10,
        dropout=0.1,
    )
    
    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(
        config,
        epochs=100,
        batch_size=settings["batch_size"],
        lr=1e-3,
        device=device,
    )
    
    logger.info("Model saved to best_tft_model.pt")
    logger.info("Ready for Phase 3: RL Agent training!")
