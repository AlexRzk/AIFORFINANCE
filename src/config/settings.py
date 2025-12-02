"""
Configuration settings for the AI Trading System.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import json


@dataclass
class ExchangeConfig:
    """Configuration for exchange connections."""
    name: str
    ws_url: str
    rest_url: str
    symbols: List[str]
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data storage and processing."""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    lob_snapshots_dir: str = "lob_snapshots"
    trades_dir: str = "trades"
    features_dir: str = "features"
    parquet_compression: str = "snappy"
    snapshot_interval_ms: int = 100  # LOB snapshot interval
    orderbook_depth: int = 20  # Number of price levels to record


@dataclass
class OFIConfig:
    """Configuration for Order Flow Imbalance calculation."""
    timeframes_ms: List[int] = field(default_factory=lambda: [100, 1000, 5000])
    rolling_window: int = 100


@dataclass
class FracDiffConfig:
    """Configuration for Fractional Differentiation."""
    d_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    adf_pvalue_threshold: float = 0.05
    min_weight_threshold: float = 1e-5  # Threshold for truncating weights


@dataclass
class Settings:
    """Main settings container."""
    # Exchange configurations
    exchanges: List[ExchangeConfig] = field(default_factory=list)
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    
    # Feature engineering configs
    ofi: OFIConfig = field(default_factory=OFIConfig)
    fracdiff: FracDiffConfig = field(default_factory=FracDiffConfig)
    
    def __post_init__(self):
        """Initialize default exchanges if none provided."""
        if not self.exchanges:
            self.exchanges = [
                ExchangeConfig(
                    name="binance",
                    ws_url="wss://stream.binance.com:9443/ws",
                    rest_url="https://api.binance.com",
                    symbols=["BTCUSDT", "ETHUSDT"]
                ),
                ExchangeConfig(
                    name="bybit",
                    ws_url="wss://stream.bybit.com/v5/public/linear",
                    rest_url="https://api.bybit.com",
                    symbols=["BTCUSDT", "ETHUSDT"]
                )
            ]
    
    def save(self, path: Path):
        """Save settings to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    def _to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "exchanges": [
                {
                    "name": e.name,
                    "ws_url": e.ws_url,
                    "rest_url": e.rest_url,
                    "symbols": e.symbols,
                }
                for e in self.exchanges
            ],
            "data": {
                "data_dir": str(self.data.data_dir),
                "lob_snapshots_dir": self.data.lob_snapshots_dir,
                "trades_dir": self.data.trades_dir,
                "features_dir": self.data.features_dir,
                "parquet_compression": self.data.parquet_compression,
                "snapshot_interval_ms": self.data.snapshot_interval_ms,
                "orderbook_depth": self.data.orderbook_depth,
            },
            "ofi": {
                "timeframes_ms": self.ofi.timeframes_ms,
                "rolling_window": self.ofi.rolling_window,
            },
            "fracdiff": {
                "d_values": self.fracdiff.d_values,
                "adf_pvalue_threshold": self.fracdiff.adf_pvalue_threshold,
                "min_weight_threshold": self.fracdiff.min_weight_threshold,
            }
        }
    
    @classmethod
    def _from_dict(cls, data: dict) -> "Settings":
        """Create from dictionary."""
        exchanges = [
            ExchangeConfig(**e) for e in data.get("exchanges", [])
        ]
        data_config = DataConfig(**{
            **data.get("data", {}),
            "data_dir": Path(data.get("data", {}).get("data_dir", "data"))
        })
        ofi_config = OFIConfig(**data.get("ofi", {}))
        fracdiff_config = FracDiffConfig(**data.get("fracdiff", {}))
        
        return cls(
            exchanges=exchanges,
            data=data_config,
            ofi=ofi_config,
            fracdiff=fracdiff_config
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
