# Training RL Agent on Real Market Data

This guide shows how to train your QR-DQN agent on real Binance cryptocurrency data.

## Local Training (with Real Data)

### Prerequisites
```bash
pip install ccxt torch gymnasium tensorboard numpy pandas
```

### Training with Binance Data

**Fetch 1 year of BTCUSDT hourly data:**
```bash
python notebooks/colab_train_rl.py --real
```

**Fetch different cryptocurrency or timeframe:**
Edit `notebooks/colab_train_rl.py` and modify parameters in the `if __name__ == "__main__"` section:
```python
symbol = "ETHUSDT"   # Change to: BNBUSDT, ADAUSDT, etc.
days = 365           # Adjust number of days
```

Then run:
```bash
python notebooks/colab_train_rl.py --real
```

### Training Workflow

```bash
# 1. Train on real data (generates best_rl_agent.pt)
python notebooks/colab_train_rl.py --real

# 2. Backtest the trained agent
python notebooks/colab_backtest_rl.py

# 3. View backtest results
# Open backtest_results.png to see performance charts
```

## Colab Training (with Real Data)

### Step 1: Clone Repository
```python
!rm -rf AIFORFINANCE
!git clone https://github.com/AlexRzk/AIFORFINANCE.git
%cd AIFORFINANCE
```

### Step 2: Install Dependencies
```python
!pip install -q ccxt torch gymnasium tensorboard
```

### Step 3: Train on Real Binance Data
```python
# Train on BTCUSDT (1 year of hourly data)
%run notebooks/colab_train_rl.py --real
```

**Expected Output:**
```
🖥️  GPU: Tesla T4 (15.8 GB)
⚙️  Optimal settings: {'batch_size': 128, 'hidden_dims': [256, 256], ...}
📍 Using device: cuda

============================================================
TRAINING WITH REAL DATA FROM BINANCE
============================================================

Fetching 365 days of BTCUSDT data from Binance...
✅ Fetched 8,760 candles for BTCUSDT
✅ Using 8,760 price points for training

[████████████████████] 100.0% - Step 100,000
✅ Training complete!
📁 Model saved to: best_rl_agent.pt
```

### Step 4: Backtest Trained Agent
```python
%run notebooks/colab_backtest_rl.py
```

**Output includes:**
- Total Return %
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside volatility)
- Max Drawdown
- Win Rate
- Backtest chart saved as `backtest_results.png`

## Available Symbols

You can train on any Binance trading pair. Popular cryptocurrencies:

| Symbol | Asset |
|--------|-------|
| BTCUSDT | Bitcoin |
| ETHUSDT | Ethereum |
| BNBUSDT | Binance Coin |
| ADAUSDT | Cardano |
| XRPUSDT | Ripple |
| DOGEUSDT | Dogecoin |
| SOLANA | Solana |
| MATICUSDT | Polygon |

To try different symbol on Colab:
```python
# Modify in colab_train_rl.py line ~660
symbol = "ETHUSDT"  # Edit here
```

## Training Performance Tips

### For Better Results:

1. **Increase Training Steps** (takes longer but better convergence):
   ```python
   agent = train_rl_agent(
       total_timesteps=500000,  # Increase from 100k
       eval_freq=50000,
       use_real_data=True,
       symbol="BTCUSDT",
   )
   ```

2. **Adjust Reward Function** in `colab_train_rl.py`:
   ```python
   dsr = DifferentialSharpeRatio(
       eta=0.01,
       drawdown_penalty=2.0,      # Higher = penalize losses more
       volatility_penalty=1.0,    # Higher = encourage smooth returns
   )
   ```

3. **Market Conditions Matter**:
   - Bull markets: Agent naturally learns to hold/buy
   - Bear markets: Agent learns to sell/avoid losses
   - Mixed markets: Most challenging - agent needs robust strategy

## Troubleshooting

### "ModuleNotFoundError: No module named 'ccxt'"
```bash
pip install ccxt
```

### "Failed to fetch real data, falling back to synthetic data"
- Check internet connection
- Binance API rate limit reached (wait 1 minute)
- Try different symbol or fewer days

### Training is slow on CPU
- Use Colab with GPU (T4, A100, or H100)
- Click: Runtime → Change runtime type → GPU
- Or reduce `total_timesteps` to 50000

### Agent performance is negative
- This is normal! Cryptocurrency is hard to predict
- Try:
  - More training steps (500k+)
  - Different symbol (ETHUSDT tends easier than BTCUSDT)
  - Adjusted reward penalties
  - Different market conditions

## Next Steps

1. **Experiment with different assets**: ETHUSDT, BNBUSDT
2. **Optimize hyperparameters**: Learning rate, network size
3. **Add TFT features**: Integrate TFT model for better context
4. **Deploy on live data**: Test on recent market conditions
5. **Ensemble methods**: Train multiple agents, vote on decisions

## Expected Sharpe Ratios

As reference (varies by market):
- **< 0.5**: Underperforming (need tuning)
- **0.5 - 1.0**: Reasonable (slightly better than random)
- **1.0 - 2.0**: Good (beating many traders)
- **> 2.0**: Excellent (very rare, check for overfitting)

---

**Questions?** Check the main README.md or GitHub Issues.
