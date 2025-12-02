"""
Training infrastructure for the TFT model.

Supports:
- GPU training on T4 (Colab), A6000, H100
- Mixed precision training (FP16/BF16)
- Gradient accumulation for limited memory
- TensorBoard logging
- Distributed training ready
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.tft import TemporalFusionTransformer, TFTConfig, TFTForecaster

logger = logging.getLogger(__name__)


@dataclass 
class TrainingConfig:
    """Training configuration."""
    
    # Data
    batch_size: int = 64
    num_workers: int = 4
    
    # Training
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    
    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 100
    
    # Device
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    
    # Distributed (for multi-GPU)
    distributed: bool = False
    
    def __post_init__(self):
        """Set device automatically."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info() -> Dict:
    """Get information about available compute devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
    
    return info


def get_optimal_settings(device_name: str) -> Dict:
    """
    Get optimal training settings based on GPU type.
    
    Args:
        device_name: GPU name (e.g., "Tesla T4", "NVIDIA A6000", "NVIDIA H100")
    
    Returns:
        Dictionary with recommended settings
    """
    device_name_lower = device_name.lower()
    
    # Default settings
    settings = {
        "batch_size": 32,
        "gradient_accumulation": 1,
        "mixed_precision": "fp16",
        "num_workers": 4,
    }
    
    if "t4" in device_name_lower:
        # Tesla T4: 16GB VRAM, good FP16 performance
        settings = {
            "batch_size": 32,
            "gradient_accumulation": 2,
            "mixed_precision": "fp16",
            "num_workers": 2,  # Colab has limited CPU
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_attention_layers": 2,
        }
    
    elif "a6000" in device_name_lower or "a100" in device_name_lower:
        # A6000: 48GB VRAM, A100: 40/80GB
        settings = {
            "batch_size": 128,
            "gradient_accumulation": 1,
            "mixed_precision": "fp16",
            "num_workers": 8,
            "embed_dim": 128,
            "hidden_dim": 128,
            "num_attention_layers": 4,
        }
    
    elif "h100" in device_name_lower:
        # H100: 80GB VRAM, excellent BF16 performance
        settings = {
            "batch_size": 256,
            "gradient_accumulation": 1,
            "mixed_precision": "bf16",  # H100 has great BF16
            "num_workers": 16,
            "embed_dim": 256,
            "hidden_dim": 256,
            "num_attention_layers": 6,
        }
    
    return settings


class TimeSeriesDataset(Dataset):
    """
    Dataset for TFT training.
    
    Handles sliding window extraction and target generation.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        encoder_length: int = 100,
        decoder_length: int = 10,
        step_size: int = 1,
    ):
        """
        Initialize dataset.
        
        Args:
            features: (num_samples, num_features) array
            targets: (num_samples,) array of values to predict
            encoder_length: Number of past timesteps
            decoder_length: Number of future timesteps to predict
            step_size: Step size for sliding window
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.step_size = step_size
        
        # Calculate valid indices
        total_length = encoder_length + decoder_length
        self.valid_indices = list(range(0, len(features) - total_length + 1, step_size))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx]
        
        # Encoder input: past features
        encoder_input = self.features[start_idx:start_idx + self.encoder_length]
        
        # Decoder target: future values
        target_start = start_idx + self.encoder_length
        target = self.targets[target_start:target_start + self.decoder_length]
        
        return encoder_input, target


class TFTTrainer:
    """
    Trainer for the Temporal Fusion Transformer.
    
    Handles:
    - Training loop with mixed precision
    - Validation and early stopping
    - Checkpointing
    - TensorBoard logging
    - GPU memory optimization
    """
    
    def __init__(
        self,
        model_config: TFTConfig,
        training_config: TrainingConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        
        # Set device
        self.device = torch.device(training_config.device)
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            device_info = get_device_info()
            if device_info["devices"]:
                gpu_name = device_info["devices"][0]["name"]
                logger.info(f"GPU: {gpu_name}")
                
                # Get optimal settings
                optimal = get_optimal_settings(gpu_name)
                logger.info(f"Recommended settings for {gpu_name}: {optimal}")
        
        # Initialize model
        self.model = TemporalFusionTransformer(model_config)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )
        
        # Mixed precision
        self.use_amp = training_config.mixed_precision != "fp32"
        if self.use_amp:
            dtype = torch.bfloat16 if training_config.mixed_precision == "bf16" else torch.float16
            self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
            self.amp_dtype = dtype
        else:
            self.scaler = None
            self.amp_dtype = torch.float32
        
        # Logging
        self.writer = None
        self.global_step = 0
        
        # Create directories
        Path(training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(training_config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
        
        Returns:
            Training history
        """
        config = self.training_config
        
        # DataLoaders - only pin memory if using CUDA
        use_pin_memory = self.device.type == 'cuda'
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size * 2,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=use_pin_memory,
            )
        
        # TensorBoard
        self.writer = SummaryWriter(config.log_dir)
        
        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {config.epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, Batch size: {config.batch_size}")
        
        for epoch in range(config.epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            
            log_msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            log_msg += f" | LR: {lr:.2e} | Time: {epoch_time:.1f}s"
            logger.info(log_msg)
            
            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("LR", lr, epoch)
            
            # Periodic checkpoint
            if (epoch + 1) % config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_loss or train_loss)
        
        self.writer.close()
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        config = self.training_config
        
        self.optimizer.zero_grad()
        
        for batch_idx, (encoder_input, target) in enumerate(train_loader):
            encoder_input = encoder_input.to(self.device)
            target = target.to(self.device)
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                losses = self.model.compute_loss(encoder_input, target)
                loss = losses["loss"] / config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % config.log_every_n_steps == 0:
                    self.writer.add_scalar(
                        "Loss/train_step",
                        loss.item() * config.gradient_accumulation_steps,
                        self.global_step
                    )
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for encoder_input, target in val_loader:
            encoder_input = encoder_input.to(self.device)
            target = target.to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                losses = self.model.compute_loss(encoder_input, target)
            
            total_loss += losses["loss"].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "loss": loss,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = Path(self.training_config.checkpoint_dir)
        
        # Save regular checkpoint
        torch.save(checkpoint, path / f"checkpoint_epoch_{epoch}.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, path / "best_model.pt")
            logger.info(f"Saved best model with loss: {loss:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint


def create_training_dataset(
    features_df: pd.DataFrame,
    target_column: str = "volatility",
    encoder_length: int = 100,
    decoder_length: int = 10,
    train_ratio: float = 0.8,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
    """
    Create training and validation datasets from a DataFrame.
    
    Args:
        features_df: DataFrame with features
        target_column: Column name for prediction target
        encoder_length: History length
        decoder_length: Prediction horizon
        train_ratio: Fraction of data for training
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Get feature columns (exclude timestamp and target)
    feature_cols = [
        col for col in features_df.columns
        if col not in ["timestamp", target_column, "symbol"]
    ]
    
    features = features_df[feature_cols].values.astype(np.float32)
    targets = features_df[target_column].values.astype(np.float32)
    
    # Replace NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train/val split
    split_idx = int(len(features) * train_ratio)
    
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    train_dataset = TimeSeriesDataset(
        train_features, train_targets,
        encoder_length, decoder_length,
    )
    
    val_dataset = TimeSeriesDataset(
        val_features, val_targets,
        encoder_length, decoder_length,
    )
    
    return train_dataset, val_dataset
