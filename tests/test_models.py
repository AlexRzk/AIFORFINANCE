"""
Tests for the TFT model and training infrastructure.
"""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tft import TFTConfig, TemporalFusionTransformer, TFTForecaster
from src.models.components import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    QuantileLoss,
)
from src.training import TimeSeriesDataset, get_device_info, get_optimal_settings


class TestGRN:
    """Test Gated Residual Network."""
    
    def test_forward(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 10, 32)
        out = grn(x)
        
        assert out.shape == (4, 10, 32)
    
    def test_with_context(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32, context_dim=16)
        x = torch.randn(4, 10, 32)
        context = torch.randn(4, 10, 16)
        out = grn(x, context)
        
        assert out.shape == (4, 10, 32)
    
    def test_dimension_change(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=64)
        x = torch.randn(4, 10, 32)
        out = grn(x)
        
        assert out.shape == (4, 10, 64)


class TestAttention:
    """Test Multi-Head Attention."""
    
    def test_forward(self):
        attn = InterpretableMultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(4, 20, 64)
        out, weights = attn(x, x, x)
        
        assert out.shape == (4, 20, 64)
        assert weights.shape == (4, 4, 20, 20)  # batch, heads, seq, seq
    
    def test_causal_masking(self):
        attn = InterpretableMultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        _, weights = attn(x, x, x)
        
        # Upper triangle should be near zero (masked)
        weights_mean = weights[0, 0].detach()
        upper = torch.triu(weights_mean, diagonal=1)
        assert upper.sum().item() < 1e-5


class TestQuantileLoss:
    """Test Quantile Loss."""
    
    def test_loss_computation(self):
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        pred = torch.randn(4, 10, 3)
        target = torch.randn(4, 10)
        
        loss = loss_fn(pred, target)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
    
    def test_quantile_ordering(self):
        """Lower quantiles should predict lower values for accurate predictions."""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        # Perfect prediction: q10 < q50 < q90
        pred = torch.tensor([[[0.1, 0.5, 0.9]]])  # Ordered quantiles
        target = torch.tensor([[0.5]])
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.5  # Should be low for good prediction


class TestTFTConfig:
    """Test TFT configuration."""
    
    def test_default_config(self):
        config = TFTConfig()
        
        assert config.num_time_varying_features == 12
        assert config.hidden_dim == 64
        assert len(config.quantiles) == config.num_quantiles
    
    def test_validation(self):
        # embed_dim must be divisible by num_heads
        config = TFTConfig(embed_dim=64, num_heads=4)
        assert config.embed_dim % config.num_heads == 0


class TestTFT:
    """Test Temporal Fusion Transformer."""
    
    @pytest.fixture
    def config(self):
        return TFTConfig(
            num_time_varying_features=12,
            hidden_dim=32,
            embed_dim=32,
            num_heads=4,
            num_attention_layers=2,
            encoder_length=50,
            decoder_length=10,
            dropout=0.1,
        )
    
    @pytest.fixture
    def model(self, config):
        return TemporalFusionTransformer(config)
    
    def test_forward(self, model, config):
        x = torch.randn(4, config.encoder_length, config.num_time_varying_features)
        out = model(x)
        
        assert out.shape == (4, config.decoder_length, config.num_quantiles)
    
    def test_forward_with_context(self, model, config):
        x = torch.randn(4, config.encoder_length, config.num_time_varying_features)
        out, context = model(x, return_context=True)
        
        assert out.shape == (4, config.decoder_length, config.num_quantiles)
        assert context.shape == (4, config.embed_dim)
    
    def test_compute_loss(self, model, config):
        x = torch.randn(4, config.encoder_length, config.num_time_varying_features)
        targets = torch.randn(4, config.decoder_length)
        
        losses = model.compute_loss(x, targets)
        
        assert "loss" in losses
        assert losses["loss"].ndim == 0
    
    def test_freeze_encoder(self, model, config):
        model.freeze_encoder()
        
        # Most params should be frozen
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        total = sum(1 for p in model.parameters())
        
        assert frozen > total * 0.9  # Most should be frozen
        
        # Context projection should be trainable
        assert model.context_projection.weight.requires_grad
    
    def test_get_context_vector(self, model, config):
        x = torch.randn(4, config.encoder_length, config.num_time_varying_features)
        context = model.get_context_vector(x)
        
        assert context.shape == (4, config.embed_dim)
    
    def test_interpretability_info(self, model, config):
        x = torch.randn(4, config.encoder_length, config.num_time_varying_features)
        _ = model(x)
        
        info = model.get_interpretability_info()
        
        assert "variable_weights" in info
        assert "attention_weights" in info
        assert info["feature_names"] == config.time_varying_feature_names


class TestDataset:
    """Test TimeSeriesDataset."""
    
    def test_creation(self):
        features = np.random.randn(1000, 12).astype(np.float32)
        targets = np.random.randn(1000).astype(np.float32)
        
        dataset = TimeSeriesDataset(features, targets, encoder_length=100, decoder_length=10)
        
        assert len(dataset) > 0
    
    def test_getitem(self):
        features = np.random.randn(500, 12).astype(np.float32)
        targets = np.random.randn(500).astype(np.float32)
        
        dataset = TimeSeriesDataset(features, targets, encoder_length=50, decoder_length=10)
        
        x, y = dataset[0]
        
        assert x.shape == (50, 12)
        assert y.shape == (10,)


class TestDeviceOptimization:
    """Test GPU detection and optimization."""
    
    def test_get_device_info(self):
        info = get_device_info()
        
        assert "cuda_available" in info
        assert "device_count" in info
        assert "devices" in info
    
    def test_get_optimal_settings(self):
        # Test T4 settings
        t4_settings = get_optimal_settings("Tesla T4")
        assert t4_settings["batch_size"] <= 64
        assert t4_settings["mixed_precision"] == "fp16"
        
        # Test H100 settings
        h100_settings = get_optimal_settings("NVIDIA H100")
        assert h100_settings["batch_size"] >= 128
        assert h100_settings["mixed_precision"] == "bf16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
