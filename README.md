# AI for Finance - Crypto Trading System

A sophisticated AI-powered cryptocurrency trading system implementing advanced machine learning techniques for high-frequency trading.

## Phase 1: Data Infrastructure (Current)

This phase implements the foundational data pipeline including:

### Features Implemented

#### 1. Data Ingestion (LOB & Ticks)
- **Binance Futures WebSocket** - Real-time Level 2 order book and aggregated trades
- **Bybit Linear WebSocket** - V5 API for USDT perpetual futures
- **High-performance recording** - Batched Parquet writes with Snappy compression

```python
from src.data_ingestion import BinanceIngestion, DataRecorder

recorder = DataRecorder("./data")
ingestion = BinanceIngestion(
    symbols=["BTCUSDT", "ETHUSDT"],
    on_orderbook=recorder.record_orderbook,
    on_trade=recorder.record_trade,
)

await asyncio.gather(recorder.start(), ingestion.start())
```

#### 2. Microstructure Feature Engineering (OFI)
- **Order Flow Imbalance** at multiple depth levels (L0, L5, L10, L20)
- **Multi-timeframe OFI** - 100ms, 1s, 5s, 30s rolling windows
- **Z-score normalization** for regime-adaptive signals
- **Volume-weighted OFI** - Distance-decay weighted imbalance

Formula: $OFI_t = \sum (Vol_{bid, i} - Vol_{ask, i})$

```python
from src.features import OrderFlowImbalance

ofi = OrderFlowImbalance(timeframes_ms=[100, 1000, 5000])
features = ofi.update(
    symbol="BTCUSDT",
    timestamp=current_time,
    bid_prices=bids,
    bid_quantities=bid_qtys,
    ask_prices=asks,
    ask_quantities=ask_qtys,
)
print(f"OFI Z-score (1s): {features.ofi_zscore_1s}")
```

#### 3. Stationarity Pipeline (FracDiff)
- **Fractional Differentiation** - Preserves memory while achieving stationarity
- **Fixed-window FFD** - Real-time compatible implementation
- **Automatic d optimization** - ADF test-based search for optimal $d$
- **Online differentiator** - Streaming FFD for live trading

Why: Standard differentiation ($d=1$) erases memory; raw prices are non-stationary. FracDiff finds the balance (e.g., $d=0.4$) to keep trend memory while making data statistically safe for ML.

```python
from src.features import find_optimal_d, FractionalDifferentiator

# Find optimal d
result = find_optimal_d(log_prices, adf_pvalue_threshold=0.05)
print(f"Optimal d: {result.d}, Memory retained: {result.memory_retained:.2%}")

# Online processing
fracdiff = FractionalDifferentiator(d=0.4)
for price in streaming_prices:
    stationary_price = fracdiff.update(np.log(price))
```

#### 4. Regime Detection
- **Fractal Dimension Index (FDI)** - Detects trending vs ranging markets
  - FDI < 1.5 → Trending market
  - FDI > 1.5 → Mean-reverting/ranging market

```python
from src.features.features import FractalDimensionIndex

fdi = FractalDimensionIndex(window=30)
for price in prices:
    regime = fdi.update(price)
    if regime < 1.5:
        print("Trending - follow momentum")
    else:
        print("Ranging - mean reversion")
```

## Project Structure

```
AIFORFINANCE/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration dataclasses
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── base.py              # Base classes, OrderBookSnapshot, Trade
│   │   ├── binance.py           # Binance WebSocket connector
│   │   ├── bybit.py             # Bybit WebSocket connector
│   │   └── recorder.py          # Parquet data storage
│   └── features/
│       ├── __init__.py
│       ├── ofi.py               # Order Flow Imbalance
│       ├── fracdiff.py          # Fractional Differentiation
│       └── features.py          # Feature engineering orchestrator
├── examples/
│   ├── record_data.py           # Live data recording example
│   ├── fracdiff_analysis.py     # FracDiff analysis example
│   └── realtime_features.py     # Real-time feature processing
├── tests/
│   ├── test_features.py
│   └── test_data_ingestion.py
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AlexRzk/AIFORFINANCE.git
cd AIFORFINANCE

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Record Market Data

```bash
python examples/record_data.py --symbols BTCUSDT ETHUSDT --duration 3600
```

### 2. Analyze Fractional Differentiation

```bash
python examples/fracdiff_analysis.py --plot
```

### 3. Real-time Feature Processing

```bash
python examples/realtime_features.py --symbol BTCUSDT --duration 60
```

## Running Tests

```bash
pytest tests/ -v
```

## Upcoming Phases

### Phase 2: AI Architecture (Weeks 5-8)
- **Temporal Fusion Transformer (TFT)** - Context encoder with attention
- **QR-DQN Agent** - Quantile regression for tail risk awareness

### Phase 3: Training Infrastructure (Weeks 9-12)
- **Differential Sharpe Ratio** reward function
- **PopArt** adaptive reward normalization
- **Combinatorial Purged Cross-Validation**

### Phase 4: Execution Layer (Weeks 13-16)
- **Monte Carlo Tree Search** for optimal order placement
- **Monte Carlo Dropout** uncertainty quantification

### Phase 5: Manager System (Weeks 17-20)
- **LLM Reflection Agent** for trade analysis
- **Vector Database** for rule storage

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*
2. Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*
3. Lim, B., et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
4. Dabney, W., et al. (2018). *Distributional Reinforcement Learning with Quantile Regression*

## License

MIT License - See LICENSE file for details.
