"""
Train TFT Model - Phase 2 Training Script.

This script:
1. Loads or generates training data
2. Configures the TFT model based on available GPU
3. Trains the model with mixed precision
4. Saves checkpoints for RL agent

Usage:
    python scripts/train_tft.py --epochs 100 --batch-size 64
    
For Colab:
    !python scripts/train_tft.py --colab
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.tft import TFTConfig, TemporalFusionTransformer
from src.training import (
    TFTTrainer,
    TrainingConfig,
    TimeSeriesDataset,
    get_device_info,
    get_optimal_settings,
    create_training_dataset,
)
from src.features.fracdiff import frac_diff_ffd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 50000,
    n_features: int = 12,
) -> pd.DataFrame:
    """
    Generate synthetic training data for development.
    
    In production, this would be replaced with real market data
    from the data ingestion pipeline.
    """
    np.random.seed(42)
    
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    # Generate realistic price series (random walk with volatility clustering)
    volatility = np.zeros(n_samples)
    volatility[0] = 0.001
    for i in range(1, n_samples):
        volatility[i] = 0.00001 + 0.1 * volatility[i-1] + 0.85 * (np.random.randn() ** 2) * 0.0001
    
    returns = np.random.randn(n_samples) * np.sqrt(volatility) + 0.00001
    log_prices = np.cumsum(returns) + np.log(50000)
    
    # Apply fractional differentiation
    log_prices_series = pd.Series(log_prices)
    price_ffd = frac_diff_ffd(log_prices_series, d=0.4).values
    
    # Pad to match length
    pad_length = n_samples - len(price_ffd)
    price_ffd = np.concatenate([np.zeros(pad_length), price_ffd])
    
    # Generate other features
    data = {
        "price_ffd": price_ffd,
        "volume_ffd": np.random.randn(n_samples) * 0.1,
        "ofi_level_0": np.random.randn(n_samples) * 10,
        "ofi_level_5": np.random.randn(n_samples) * 50,
        "ofi_1s": np.cumsum(np.random.randn(n_samples) * 0.1),
        "ofi_5s": np.cumsum(np.random.randn(n_samples) * 0.05),
        "ofi_zscore_1s": np.random.randn(n_samples),
        "spread": np.abs(np.random.randn(n_samples) * 0.1) + 0.01,
        "trade_imbalance": np.random.randn(n_samples) * 100,
        "fdi": 1.5 + np.random.randn(n_samples) * 0.2,
        "volatility": volatility,
        "global_risk_score": np.clip(0.5 + np.random.randn(n_samples) * 0.1, 0, 1),
    }
    
    # Target: future volatility (10-step ahead)
    data["target_volatility"] = np.roll(volatility, -10)
    data["target_volatility"][-10:] = volatility[-10:]  # Handle edge
    
    df = pd.DataFrame(data)
    
    # Normalize features
    for col in df.columns:
        if col != "target_volatility":
            mean = df[col].mean()
            std = df[col].std() + 1e-8
            df[col] = (df[col] - mean) / std
    
    logger.info(f"Generated DataFrame with shape {df.shape}")
    
    return df


def main(args):
    """Main training function."""
    
    # Check GPU
    logger.info("=" * 60)
    logger.info("GPU INFORMATION")
    logger.info("=" * 60)
    
    device_info = get_device_info()
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info["devices"]:
        gpu = device_info["devices"][0]
        logger.info(f"GPU: {gpu['name']}")
        logger.info(f"Memory: {gpu['total_memory_gb']:.1f} GB")
        logger.info(f"Compute Capability: {gpu['compute_capability']}")
        
        # Get optimal settings
        optimal = get_optimal_settings(gpu["name"])
        logger.info(f"Optimal settings: {optimal}")
        
        # Apply optimal settings if --auto
        if args.auto_config:
            args.batch_size = optimal.get("batch_size", args.batch_size)
            args.embed_dim = optimal.get("embed_dim", args.embed_dim)
            args.hidden_dim = optimal.get("hidden_dim", args.hidden_dim)
            args.mixed_precision = optimal.get("mixed_precision", args.mixed_precision)
    
    # Create config
    model_config = TFTConfig(
        num_time_varying_features=12,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        lstm_layers=args.lstm_layers,
        num_heads=args.num_heads,
        num_attention_layers=args.num_attention_layers,
        encoder_length=args.encoder_length,
        decoder_length=args.decoder_length,
        dropout=args.dropout,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    
    logger.info("=" * 60)
    logger.info("MODEL CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Embed dim: {model_config.embed_dim}")
    logger.info(f"Hidden dim: {model_config.hidden_dim}")
    logger.info(f"Attention layers: {model_config.num_attention_layers}")
    logger.info(f"Encoder length: {model_config.encoder_length}")
    logger.info(f"Decoder length: {model_config.decoder_length}")
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"Epochs: {training_config.epochs}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Mixed precision: {training_config.mixed_precision}")
    logger.info(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    
    # Load or generate data
    if args.data_path and Path(args.data_path).exists():
        data_path = Path(args.data_path)
        
        if data_path.is_dir():
            # Directory with Parquet files - use MarketDataLoader
            logger.info(f"Loading real market data from {args.data_path}")
            from src.data import MarketDataLoader, DataConfig
            
            data_config = DataConfig(
                data_dir=str(data_path),
                encoder_length=model_config.encoder_length,
                decoder_length=model_config.decoder_length,
            )
            loader = MarketDataLoader(data_config)
            train_dataset, val_dataset, _ = loader.prepare_datasets(
                target_type="returns"
            )
            # Skip the create_training_dataset call below
            df = None
        else:
            # Single Parquet file
            logger.info(f"Loading data from {args.data_path}")
            df = pd.read_parquet(args.data_path)
    else:
        logger.info("No data path provided, generating synthetic data...")
        logger.info("Use --data-path <dir> to load real market data from Parquet files")
        df = generate_synthetic_data(n_samples=args.n_samples)
    
    # Create datasets (only if using synthetic data or single file)
    if df is not None:
        train_dataset, val_dataset = create_training_dataset(
            df,
            target_column="target_volatility",
            encoder_length=model_config.encoder_length,
            decoder_length=model_config.decoder_length,
            train_ratio=0.8,
        )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = TFTTrainer(model_config, training_config)
    
    # Model summary
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    history = trainer.train(train_dataset, val_dataset)
    
    # Save final model
    final_path = Path(training_config.checkpoint_dir) / "final_model.pt"
    trainer._save_checkpoint(args.epochs - 1, history["val_loss"][-1] if history["val_loss"] else history["train_loss"][-1])
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best val loss: {min(history['val_loss']) if history['val_loss'] else 'N/A'}")
    logger.info(f"Checkpoints saved to: {training_config.checkpoint_dir}")
    logger.info(f"Logs saved to: {training_config.log_dir}")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT Model")
    
    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to training data (parquet)")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Number of synthetic samples if no data")
    
    # Model architecture
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-attention-layers", type=int, default=2)
    parser.add_argument("--encoder-length", type=int, default=100)
    parser.add_argument("--decoder-length", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    
    # Auto-configuration
    parser.add_argument("--auto-config", action="store_true",
                        help="Auto-configure based on GPU")
    parser.add_argument("--colab", action="store_true",
                        help="Optimize for Google Colab T4")
    
    args = parser.parse_args()
    
    # Colab presets
    if args.colab:
        args.auto_config = True
        args.num_workers = 2  # Colab has limited CPU
        args.batch_size = 32
        args.gradient_accumulation = 2
    
    main(args)
