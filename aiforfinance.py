"""
AI for Finance - Phase 1: Data Infrastructure

This module provides:
1. Real-time data ingestion from Binance/Bybit WebSockets
2. Order Flow Imbalance (OFI) feature engineering
3. Fractional Differentiation for stationarity
4. Parquet-based data storage

Usage:
    from aiforfinance import DataRecorder, BinanceIngestion, FeatureEngineer

    # Start data recording
    recorder = DataRecorder("./data")
    ingestion = BinanceIngestion(["BTCUSDT"], on_orderbook=recorder.record_orderbook)
    
    await asyncio.gather(recorder.start(), ingestion.start())
"""

from src.config import Settings, get_settings
from src.data_ingestion import (
    BinanceIngestion,
    BybitIngestion,
    DataRecorder,
    OrderBookSnapshot,
    Trade,
)
from src.features import (
    OrderFlowImbalance,
    OFIFeatures,
    FractionalDifferentiator,
    find_optimal_d,
    FeatureEngineer,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Data Ingestion
    "BinanceIngestion",
    "BybitIngestion",
    "DataRecorder",
    "OrderBookSnapshot",
    "Trade",
    # Features
    "OrderFlowImbalance",
    "OFIFeatures",
    "FractionalDifferentiator",
    "find_optimal_d",
    "FeatureEngineer",
]
