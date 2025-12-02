"""
Data loader for connecting real market data to the TFT model.

This module loads Parquet files created by the data ingestion pipeline
and prepares them for training/inference.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from src.features.fracdiff import FractionalDifferentiator, find_optimal_d
from src.features.ofi import OrderFlowImbalance
from src.features.features import FractalDimensionIndex

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    # Paths
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    
    # Feature engineering
    fracdiff_d: float = 0.4
    ofi_depths: List[int] = field(default_factory=lambda: [0, 5, 10, 20])
    ofi_windows_ms: List[int] = field(default_factory=lambda: [100, 1000, 5000])
    fdi_window: int = 50
    
    # Sequence parameters
    encoder_length: int = 100
    decoder_length: int = 10
    
    # Normalization
    normalize: bool = True
    clip_outliers: float = 5.0  # Clip to N std deviations
    
    # Train/val split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class MarketDataLoader:
    """
    Load and process market data from Parquet files.
    
    Connects Phase 1 (data ingestion) to Phase 2 (TFT model).
    """
    
    # Standard feature names for TFT
    FEATURE_COLUMNS = [
        "price_ffd",       # Fractionally differenced log price
        "volume_ffd",      # Fractionally differenced volume
        "ofi_L0",          # Order Flow Imbalance at best bid/ask
        "ofi_L5",          # OFI at 5th level
        "ofi_L10",         # OFI at 10th level  
        "ofi_L20",         # OFI at 20th level
        "ofi_100ms",       # OFI over 100ms window
        "ofi_1s",          # OFI over 1s window
        "ofi_5s",          # OFI over 5s window
        "spread",          # Bid-ask spread
        "fdi",             # Fractal Dimension Index
        "volatility",      # Rolling volatility
    ]
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.fracdiff = FractionalDifferentiator(d=config.fracdiff_d)
        self.ofi = OrderFlowImbalance(depths=config.ofi_depths)
        self.fdi = FractalDimensionIndex(window=config.fdi_window)
        
        # Normalization stats (computed from training data)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
    
    def load_parquet_files(
        self,
        data_dir: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load order book and trade data from Parquet files.
        
        Args:
            data_dir: Directory containing Parquet files
            symbols: List of symbols to load (e.g., ["BTCUSDT", "ETHUSDT"])
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
        
        Returns:
            DataFrame with raw market data
        """
        data_path = Path(data_dir or self.config.data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Find all Parquet files
        parquet_files = list(data_path.glob("**/*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {data_path}")
        
        logger.info(f"Found {len(parquet_files)} Parquet files")
        
        # Filter by symbol if specified
        if symbols:
            parquet_files = [
                f for f in parquet_files 
                if any(s.lower() in f.stem.lower() for s in symbols)
            ]
        
        # Load and concatenate
        dfs = []
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                
                # Filter by date if specified
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    if start_date:
                        df = df[df["timestamp"] >= start_date]
                    if end_date:
                        df = df[df["timestamp"] <= end_date]
                
                dfs.append(df)
                logger.debug(f"Loaded {len(df)} rows from {pf.name}")
            except Exception as e:
                logger.warning(f"Failed to load {pf}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded from Parquet files")
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined)} total rows")
        return combined
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from raw market data.
        
        Expected columns in df:
            - mid_price or price
            - volume (optional)
            - bid_prices, bid_sizes, ask_prices, ask_sizes (for OFI)
            - spread (optional, computed if not present)
        
        Returns:
            DataFrame with computed features
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Price features with FracDiff
        price_col = "mid_price" if "mid_price" in df.columns else "price"
        if price_col in df.columns:
            log_price = np.log(df[price_col].replace(0, np.nan).ffill())
            
            # Apply fractional differentiation
            price_ffd = self.fracdiff.transform(log_price.values)
            features["price_ffd"] = price_ffd
        else:
            features["price_ffd"] = 0.0
        
        # 2. Volume features with FracDiff
        if "volume" in df.columns:
            log_volume = np.log1p(df["volume"].fillna(0))
            volume_ffd = self.fracdiff.transform(log_volume.values)
            features["volume_ffd"] = volume_ffd
        else:
            features["volume_ffd"] = 0.0
        
        # 3. Order Flow Imbalance
        if all(c in df.columns for c in ["bid_prices", "bid_sizes", "ask_prices", "ask_sizes"]):
            ofi_features = self._compute_ofi(df)
            for col in ofi_features.columns:
                features[col] = ofi_features[col]
        else:
            # Fill with zeros if orderbook data not available
            for depth in self.config.ofi_depths:
                features[f"ofi_L{depth}"] = 0.0
            features["ofi_100ms"] = 0.0
            features["ofi_1s"] = 0.0
            features["ofi_5s"] = 0.0
        
        # 4. Spread
        if "spread" in df.columns:
            features["spread"] = df["spread"]
        elif "best_ask" in df.columns and "best_bid" in df.columns:
            features["spread"] = df["best_ask"] - df["best_bid"]
        else:
            features["spread"] = 0.0
        
        # 5. Fractal Dimension Index
        if price_col in df.columns:
            fdi_values = self._compute_fdi(df[price_col].values)
            features["fdi"] = fdi_values
        else:
            features["fdi"] = 1.5  # Neutral FDI
        
        # 6. Volatility (rolling std of returns)
        if price_col in df.columns:
            returns = df[price_col].pct_change().fillna(0)
            features["volatility"] = returns.rolling(window=50, min_periods=1).std().fillna(0)
        else:
            features["volatility"] = 0.0
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def _compute_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Order Flow Imbalance features."""
        ofi_data = pd.DataFrame(index=df.index)
        
        # Initialize OFI calculator
        ofi = OrderFlowImbalance(depths=self.config.ofi_depths)
        
        ofi_L0 = []
        ofi_L5 = []
        ofi_L10 = []
        ofi_L20 = []
        
        for idx, row in df.iterrows():
            try:
                # Parse orderbook levels
                bid_prices = self._parse_array(row["bid_prices"])
                bid_sizes = self._parse_array(row["bid_sizes"])
                ask_prices = self._parse_array(row["ask_prices"])
                ask_sizes = self._parse_array(row["ask_sizes"])
                
                # Compute instant OFI at different depths
                ofi_values = ofi.compute_instant_ofi(
                    bid_prices, bid_sizes, ask_prices, ask_sizes
                )
                
                ofi_L0.append(ofi_values.get(0, 0))
                ofi_L5.append(ofi_values.get(5, 0))
                ofi_L10.append(ofi_values.get(10, 0))
                ofi_L20.append(ofi_values.get(20, 0))
            except Exception:
                ofi_L0.append(0)
                ofi_L5.append(0)
                ofi_L10.append(0)
                ofi_L20.append(0)
        
        ofi_data["ofi_L0"] = ofi_L0
        ofi_data["ofi_L5"] = ofi_L5
        ofi_data["ofi_L10"] = ofi_L10
        ofi_data["ofi_L20"] = ofi_L20
        
        # Compute rolling OFI windows
        ofi_data["ofi_100ms"] = ofi_data["ofi_L0"].rolling(10, min_periods=1).mean()
        ofi_data["ofi_1s"] = ofi_data["ofi_L0"].rolling(100, min_periods=1).mean()
        ofi_data["ofi_5s"] = ofi_data["ofi_L0"].rolling(500, min_periods=1).mean()
        
        return ofi_data
    
    def _compute_fdi(self, prices: np.ndarray) -> np.ndarray:
        """Compute Fractal Dimension Index."""
        fdi_values = np.full(len(prices), 1.5)  # Default neutral
        
        window = self.config.fdi_window
        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]
            
            if len(window_prices) >= 2:
                # Simple box-counting approximation
                price_range = np.max(window_prices) - np.min(window_prices)
                if price_range > 0:
                    # Normalized range
                    fdi_values[i] = 1.0 + np.log(price_range / np.std(window_prices + 1e-8)) / np.log(2)
        
        return np.clip(fdi_values, 1.0, 2.0)
    
    @staticmethod
    def _parse_array(value) -> np.ndarray:
        """Parse array from various formats."""
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, list):
            return np.array(value)
        elif isinstance(value, str):
            # Try to parse string representation
            import ast
            return np.array(ast.literal_eval(value))
        else:
            return np.array([])
    
    def normalize_features(
        self,
        features: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: DataFrame with feature columns
            fit: If True, compute and store normalization stats
        
        Returns:
            Normalized DataFrame
        """
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in features.columns]
        
        if fit or self.feature_means is None:
            self.feature_means = features[feature_cols].mean().values
            self.feature_stds = features[feature_cols].std().values
            self.feature_stds[self.feature_stds < 1e-8] = 1.0  # Avoid division by zero
        
        normalized = features.copy()
        for i, col in enumerate(feature_cols):
            normalized[col] = (features[col] - self.feature_means[i]) / self.feature_stds[i]
            
            # Clip outliers
            if self.config.clip_outliers > 0:
                normalized[col] = normalized[col].clip(
                    -self.config.clip_outliers, 
                    self.config.clip_outliers
                )
        
        return normalized
    
    def create_target(
        self,
        df: pd.DataFrame,
        target_type: str = "returns",
        horizon: int = 10,
    ) -> np.ndarray:
        """
        Create prediction targets.
        
        Args:
            df: DataFrame with price data
            target_type: Type of target ("returns", "volatility", "direction")
            horizon: Prediction horizon in timesteps
        
        Returns:
            Target array
        """
        price_col = "mid_price" if "mid_price" in df.columns else "price"
        
        if price_col not in df.columns:
            # Generate dummy target
            return np.zeros(len(df))
        
        prices = df[price_col].values
        
        if target_type == "returns":
            # Future log returns
            future_prices = np.roll(prices, -horizon)
            future_prices[-horizon:] = prices[-horizon:]
            targets = np.log(future_prices / prices)
            
        elif target_type == "volatility":
            # Future realized volatility
            returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
            targets = pd.Series(returns).rolling(horizon).std().shift(-horizon).fillna(0).values
            
        elif target_type == "direction":
            # Future price direction (up=1, down=0)
            future_prices = np.roll(prices, -horizon)
            future_prices[-horizon:] = prices[-horizon:]
            targets = (future_prices > prices).astype(float)
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return targets
    
    def prepare_datasets(
        self,
        data_dir: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        target_type: str = "returns",
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load data and create train/val/test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load raw data
        df = self.load_parquet_files(data_dir, symbols)
        
        # Compute features
        features = self.compute_features(df)
        
        # Create targets
        targets = self.create_target(df, target_type, self.config.decoder_length)
        
        # Split data
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        train_features = features.iloc[:train_end]
        val_features = features.iloc[train_end:val_end]
        test_features = features.iloc[val_end:]
        
        train_targets = targets[:train_end]
        val_targets = targets[train_end:val_end]
        test_targets = targets[val_end:]
        
        # Normalize (fit on training data only)
        train_features = self.normalize_features(train_features, fit=True)
        val_features = self.normalize_features(val_features, fit=False)
        test_features = self.normalize_features(test_features, fit=False)
        
        # Convert to numpy
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in features.columns]
        
        train_X = train_features[feature_cols].values.astype(np.float32)
        val_X = val_features[feature_cols].values.astype(np.float32)
        test_X = test_features[feature_cols].values.astype(np.float32)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_X, train_targets,
            self.config.encoder_length, self.config.decoder_length
        )
        val_dataset = TimeSeriesDataset(
            val_X, val_targets,
            self.config.encoder_length, self.config.decoder_length
        )
        test_dataset = TimeSeriesDataset(
            test_X, test_targets,
            self.config.encoder_length, self.config.decoder_length
        )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        encoder_length: int = 100,
        decoder_length: int = 10,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        
        total_length = encoder_length + decoder_length
        self.valid_indices = [
            i for i in range(len(features) - total_length + 1)
        ]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        
        x = self.features[start:start + self.encoder_length]
        y = self.targets[start + self.encoder_length:start + self.encoder_length + self.decoder_length]
        
        return x, y


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for training."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader, test_loader
