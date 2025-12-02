"""
Example: Fractional Differentiation Analysis

This script demonstrates how to find the optimal fractional
differentiation order (d) for price data.

Usage:
    python examples/fracdiff_analysis.py --data data/sample_prices.csv
"""
import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.fracdiff import (
    find_optimal_d,
    frac_diff_ffd,
    adf_test,
    plot_fracdiff_analysis,
    apply_fracdiff_to_ohlcv,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n: int = 5000) -> pd.DataFrame:
    """Generate sample price data for demonstration."""
    np.random.seed(42)
    
    # Generate random walk with drift (non-stationary)
    returns = np.random.randn(n) * 0.001 + 0.00001  # Small positive drift
    log_prices = np.cumsum(returns) + np.log(100)  # Start at $100
    prices = np.exp(log_prices)
    
    # Generate timestamps
    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="1min")
    
    # Create OHLCV
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.001)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.001)),
        "close": prices * (1 + np.random.randn(n) * 0.0005),
        "volume": np.abs(np.random.randn(n) * 1000) + 100,
    })
    
    return df


def main(data_path: str = None, save_plot: bool = False):
    """
    Run fractional differentiation analysis.
    
    Args:
        data_path: Path to CSV with price data (optional)
        save_plot: Whether to save the analysis plot
    """
    # Load or generate data
    if data_path and Path(data_path).exists():
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        if "close" in df.columns:
            prices = df["close"]
        elif "price" in df.columns:
            prices = df["price"]
        else:
            prices = df.iloc[:, 0]  # Use first column
    else:
        logger.info("Generating sample price data...")
        df = generate_sample_data(5000)
        prices = df["close"]
    
    logger.info(f"Data shape: {len(prices)} observations")
    
    # Test stationarity of raw prices
    logger.info("\n=== Testing Raw Prices ===")
    adf_stat, adf_pvalue, critical_values = adf_test(prices)
    logger.info(f"ADF Statistic: {adf_stat:.4f}")
    logger.info(f"P-value: {adf_pvalue:.4f}")
    logger.info(f"Critical Values: {critical_values}")
    logger.info(f"Stationary: {adf_pvalue < 0.05}")
    
    # Test first-difference (returns)
    logger.info("\n=== Testing First Difference (d=1) ===")
    returns = np.log(prices).diff().dropna()
    adf_stat, adf_pvalue, _ = adf_test(returns)
    logger.info(f"ADF Statistic: {adf_stat:.4f}")
    logger.info(f"P-value: {adf_pvalue:.4f}")
    logger.info(f"Stationary: {adf_pvalue < 0.05}")
    corr = np.corrcoef(prices[1:].values, returns.values)[0, 1]
    logger.info(f"Correlation with original: {corr:.4f} (memory lost!)")
    
    # Find optimal d
    logger.info("\n=== Finding Optimal d ===")
    log_prices = np.log(prices)
    log_prices.index = pd.RangeIndex(len(log_prices))
    
    result = find_optimal_d(
        log_prices,
        d_values=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        adf_pvalue_threshold=0.05,
    )
    
    logger.info(f"\n=== Results ===")
    logger.info(f"Optimal d: {result.d:.2f}")
    logger.info(f"ADF Statistic: {result.adf_stat:.4f}")
    logger.info(f"ADF P-value: {result.adf_pvalue:.4f}")
    logger.info(f"Is Stationary: {result.is_stationary}")
    logger.info(f"Memory Retained: {result.memory_retained:.4f}")
    
    # Compare d values
    logger.info("\n=== Memory vs Stationarity Trade-off ===")
    d_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    for d in d_values:
        diff_series = frac_diff_ffd(log_prices, d)
        if len(diff_series) < 20:
            continue
            
        adf_stat, adf_pvalue, _ = adf_test(diff_series)
        
        overlap = diff_series.index.intersection(log_prices.index)
        corr = np.corrcoef(log_prices.loc[overlap], diff_series.loc[overlap])[0, 1]
        
        logger.info(f"d={d:.1f}: ADF p={adf_pvalue:.4f}, memory={corr:.4f}, stationary={adf_pvalue < 0.05}")
    
    # Apply to OHLCV
    logger.info("\n=== Applying to OHLCV DataFrame ===")
    result_df, optimal_ds = apply_fracdiff_to_ohlcv(
        df,
        columns=["close", "volume"],
        adf_threshold=0.05,
    )
    
    logger.info(f"Optimal d values: {optimal_ds}")
    logger.info(f"Result columns: {list(result_df.columns)}")
    
    # Save results
    output_path = Path("data/fracdiff_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")
    
    # Plot analysis
    if save_plot:
        try:
            fig = plot_fracdiff_analysis(log_prices)
            fig.savefig("data/fracdiff_analysis.png", dpi=150, bbox_inches="tight")
            logger.info("Saved plot to data/fracdiff_analysis.png")
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fractional differentiation analysis")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file with price data"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save analysis plot"
    )
    
    args = parser.parse_args()
    main(data_path=args.data, save_plot=args.plot)
