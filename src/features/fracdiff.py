"""
Fractional Differentiation (FracDiff) for Stationary Time Series.

Standard differentiation (returns) erases memory; raw prices are non-stationary.
Fractional differentiation finds the balance (e.g., d=0.4) to keep trend memory
while making data statistically safe for ML.

References:
- "Advances in Financial Machine Learning" (Marcos López de Prado, 2018)
- Chapter 5: Fractionally Differentiated Features

The key insight is that integer differentiation (d=1) removes too much memory,
while d=0 (raw prices) is non-stationary. FracDiff with d ∈ (0, 1) provides
the best of both worlds.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FracDiffResult:
    """Result of fractional differentiation analysis."""
    d: float  # Optimal d value
    series: pd.Series  # The differentiated series
    adf_stat: float  # ADF test statistic
    adf_pvalue: float  # ADF p-value
    adf_critical_values: Dict[str, float]  # Critical values
    memory_retained: float  # Fraction of original memory retained
    is_stationary: bool  # Whether the series is stationary


def get_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute the weights for fractional differentiation.
    
    The weights follow the binomial series expansion:
    w_k = -w_{k-1} * (d - k + 1) / k
    
    Args:
        d: Fractional differentiation order (0 < d < 1 typically)
        size: Maximum number of weights
        threshold: Minimum weight magnitude to keep
    
    Returns:
        Array of weights (truncated at threshold)
    """
    weights = [1.0]
    k = 1
    
    while k < size:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    
    return np.array(weights[::-1])  # Reverse for convolution


def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    """
    Compute weights for Fixed-window Fractional Differentiation (FFD).
    
    FFD uses a fixed window of weights, making it more practical for
    real-time applications where you can't wait for all history.
    
    Args:
        d: Fractional differentiation order
        threshold: Weight cutoff threshold
        max_size: Maximum window size
    
    Returns:
        Array of weights
    """
    weights = [1.0]
    k = 1
    
    while k < max_size:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    
    return np.array(weights[::-1])


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply fractional differentiation to a time series.
    
    This is the standard (expanding window) implementation where each
    point uses all available history.
    
    Args:
        series: Input time series (e.g., log prices)
        d: Differentiation order (0 < d < 1 for partial memory)
        threshold: Weight cutoff threshold
    
    Returns:
        Fractionally differentiated series
    """
    weights = get_weights(d, len(series), threshold)
    width = len(weights)
    
    # Apply weights using convolution
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(width - 1, len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i - width + 1:i + 1].values)
    
    return result.dropna()


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply Fixed-window Fractional Differentiation (FFD).
    
    FFD is more practical than standard frac_diff because:
    1. It uses a fixed window size (determined by threshold)
    2. It can be computed in real-time without full history
    3. It's more numerically stable
    
    Args:
        series: Input time series
        d: Differentiation order
        threshold: Weight cutoff threshold
    
    Returns:
        FFD transformed series
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    if width > len(series):
        logger.warning(f"Series length ({len(series)}) < weight window ({width})")
        return pd.Series(dtype=float)
    
    # Use numpy for efficient convolution
    result = np.convolve(series.values, weights, mode='valid')
    
    # Align with original index
    return pd.Series(
        result,
        index=series.index[width - 1:],
        name=f"{series.name}_d{d}" if series.name else f"ffd_d{d}"
    )


def adf_test(series: pd.Series, regression: str = "c") -> Tuple[float, float, Dict[str, float]]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        regression: Type of regression ("c" = constant, "ct" = constant + trend)
    
    Returns:
        Tuple of (test_statistic, p_value, critical_values)
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Remove NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < 20:
        logger.warning("Series too short for ADF test")
        return np.nan, 1.0, {}
    
    try:
        result = adfuller(clean_series, regression=regression, autolag='AIC')
        return result[0], result[1], result[4]
    except Exception as e:
        logger.error(f"ADF test failed: {e}")
        return np.nan, 1.0, {}


def find_optimal_d(
    series: pd.Series,
    d_values: List[float] = None,
    threshold: float = 1e-5,
    adf_pvalue_threshold: float = 0.05,
    use_ffd: bool = True,
) -> FracDiffResult:
    """
    Find the optimal fractional differentiation order.
    
    Iterates through d values to find the minimum d that makes
    the series stationary (ADF p-value < threshold).
    
    Args:
        series: Input time series (should be log prices or similar)
        d_values: List of d values to test (default: 0.1 to 1.0)
        threshold: Weight cutoff threshold for FracDiff
        adf_pvalue_threshold: P-value threshold for stationarity
        use_ffd: Whether to use FFD (recommended)
    
    Returns:
        FracDiffResult with optimal d and transformed series
    """
    if d_values is None:
        d_values = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    logger.info(f"Finding optimal d from {d_values}")
    
    # Test each d value
    results = []
    
    for d in d_values:
        if use_ffd:
            diff_series = frac_diff_ffd(series, d, threshold)
        else:
            diff_series = frac_diff(series, d, threshold)
        
        if len(diff_series) < 20:
            continue
        
        adf_stat, adf_pvalue, critical_values = adf_test(diff_series)
        
        # Calculate memory retained (correlation with original)
        overlap_idx = diff_series.index.intersection(series.index)
        if len(overlap_idx) > 0:
            orig_subset = series.loc[overlap_idx]
            diff_subset = diff_series.loc[overlap_idx]
            memory_retained = np.corrcoef(orig_subset, diff_subset)[0, 1]
        else:
            memory_retained = 0.0
        
        is_stationary = adf_pvalue < adf_pvalue_threshold
        
        results.append({
            "d": d,
            "series": diff_series,
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "critical_values": critical_values,
            "memory_retained": memory_retained,
            "is_stationary": is_stationary,
        })
        
        logger.info(
            f"d={d:.2f}: ADF stat={adf_stat:.4f}, p-value={adf_pvalue:.4f}, "
            f"memory={memory_retained:.4f}, stationary={is_stationary}"
        )
        
        # Stop at first stationary d (minimum d for stationarity)
        if is_stationary:
            break
    
    if not results:
        logger.error("No valid results found")
        return FracDiffResult(
            d=1.0,
            series=series.diff().dropna(),
            adf_stat=np.nan,
            adf_pvalue=1.0,
            adf_critical_values={},
            memory_retained=0.0,
            is_stationary=False,
        )
    
    # Select optimal d (first stationary, or lowest p-value)
    stationary_results = [r for r in results if r["is_stationary"]]
    
    if stationary_results:
        # Choose the lowest d that achieves stationarity
        optimal = stationary_results[0]
    else:
        # No stationary result found, choose lowest p-value
        optimal = min(results, key=lambda x: x["adf_pvalue"])
        logger.warning(f"No stationary d found, using d={optimal['d']} with lowest p-value")
    
    return FracDiffResult(
        d=optimal["d"],
        series=optimal["series"],
        adf_stat=optimal["adf_stat"],
        adf_pvalue=optimal["adf_pvalue"],
        adf_critical_values=optimal["critical_values"],
        memory_retained=optimal["memory_retained"],
        is_stationary=optimal["is_stationary"],
    )


class FractionalDifferentiator:
    """
    Online fractional differentiator for real-time processing.
    
    Maintains a rolling buffer to compute FFD in streaming fashion.
    """
    
    def __init__(
        self,
        d: float,
        threshold: float = 1e-5,
        max_window: int = 10000,
    ):
        """
        Initialize the online differentiator.
        
        Args:
            d: Fractional differentiation order
            threshold: Weight cutoff threshold
            max_window: Maximum buffer size
        """
        self.d = d
        self.threshold = threshold
        self.max_window = max_window
        
        # Compute weights
        self.weights = get_weights_ffd(d, threshold, max_window)
        self.window_size = len(self.weights)
        
        # Rolling buffer
        self._buffer: List[float] = []
    
    def update(self, value: float) -> Optional[float]:
        """
        Add a new value and compute the FFD output.
        
        Args:
            value: New observation
        
        Returns:
            FFD value if buffer is full, None otherwise
        """
        self._buffer.append(value)
        
        # Trim buffer to max window
        if len(self._buffer) > self.max_window:
            self._buffer = self._buffer[-self.max_window:]
        
        # Need enough history
        if len(self._buffer) < self.window_size:
            return None
        
        # Compute weighted sum
        values = np.array(self._buffer[-self.window_size:])
        return np.dot(self.weights, values)
    
    def reset(self):
        """Clear the buffer."""
        self._buffer = []
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)


def apply_fracdiff_to_ohlcv(
    df: pd.DataFrame,
    d: float = None,
    columns: List[str] = None,
    adf_threshold: float = 0.05,
    weight_threshold: float = 1e-5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply fractional differentiation to OHLCV data.
    
    Automatically finds optimal d for each column if not specified.
    
    Args:
        df: DataFrame with OHLCV columns
        d: Specific d to use (if None, finds optimal for each column)
        columns: Columns to transform (default: OHLCV + Volume)
        adf_threshold: P-value threshold for stationarity
        weight_threshold: Weight cutoff for FFD
    
    Returns:
        Tuple of (transformed DataFrame, dict of optimal d values per column)
    """
    if columns is None:
        columns = ["open", "high", "low", "close", "volume"]
        columns = [c for c in columns if c in df.columns]
    
    result = df.copy()
    optimal_ds = {}
    
    for col in columns:
        series = df[col].dropna()
        
        # Use log for prices, raw for volume
        if col != "volume":
            series = np.log(series)
        
        if d is not None:
            # Use specified d
            diff_series = frac_diff_ffd(series, d, weight_threshold)
            optimal_ds[col] = d
        else:
            # Find optimal d
            frac_result = find_optimal_d(
                series,
                adf_pvalue_threshold=adf_threshold,
                threshold=weight_threshold,
            )
            diff_series = frac_result.series
            optimal_ds[col] = frac_result.d
            
            logger.info(
                f"Column '{col}': optimal d={frac_result.d:.2f}, "
                f"memory={frac_result.memory_retained:.4f}"
            )
        
        result[f"{col}_ffd"] = diff_series
    
    return result.dropna(), optimal_ds


def plot_fracdiff_analysis(
    series: pd.Series,
    d_values: List[float] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot fractional differentiation analysis.
    
    Shows:
    1. Original vs differentiated series at different d values
    2. ADF statistic vs d
    3. Memory (correlation) vs d
    
    Args:
        series: Original time series
        d_values: d values to analyze
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    if d_values is None:
        d_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    adf_stats = []
    adf_pvalues = []
    correlations = []
    
    # Plot original series
    axes[0, 0].plot(series.index, series.values, label="Original (d=0)")
    axes[0, 0].set_title("Time Series at Different d Values")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].legend()
    
    for d in d_values:
        if d == 0:
            diff_series = series
        elif d == 1:
            diff_series = series.diff().dropna()
        else:
            diff_series = frac_diff_ffd(series, d)
        
        if len(diff_series) < 20:
            continue
        
        # ADF test
        adf_stat, adf_pvalue, _ = adf_test(diff_series)
        adf_stats.append(adf_stat)
        adf_pvalues.append(adf_pvalue)
        
        # Correlation with original
        overlap = diff_series.index.intersection(series.index)
        if len(overlap) > 0:
            corr = np.corrcoef(
                series.loc[overlap],
                diff_series.loc[overlap]
            )[0, 1]
        else:
            corr = 0
        correlations.append(corr)
    
    # Plot ADF statistic vs d
    axes[0, 1].plot(d_values[:len(adf_stats)], adf_stats, 'b-o')
    axes[0, 1].axhline(-2.86, color='r', linestyle='--', label='5% critical value')
    axes[0, 1].set_xlabel("d")
    axes[0, 1].set_ylabel("ADF Statistic")
    axes[0, 1].set_title("ADF Statistic vs d")
    axes[0, 1].legend()
    
    # Plot p-value vs d
    axes[1, 0].plot(d_values[:len(adf_pvalues)], adf_pvalues, 'g-o')
    axes[1, 0].axhline(0.05, color='r', linestyle='--', label='5% significance')
    axes[1, 0].set_xlabel("d")
    axes[1, 0].set_ylabel("ADF p-value")
    axes[1, 0].set_title("ADF p-value vs d")
    axes[1, 0].legend()
    
    # Plot correlation vs d
    axes[1, 1].plot(d_values[:len(correlations)], correlations, 'm-o')
    axes[1, 1].set_xlabel("d")
    axes[1, 1].set_ylabel("Correlation with Original")
    axes[1, 1].set_title("Memory Retention vs d")
    
    plt.tight_layout()
    return fig
