import pandas as pd
from shared.numpy_utils import np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# DEPRECATED: Используйте shared.fibonacci_tools
# from shared.fibonacci_tools import calculate_fibonacci_levels


@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels"""

    retracement: Dict[float, float]
    extension: Dict[float, float]
    high: float
    low: float
    high_idx: int
    low_idx: int


def find_swing_points(
    high: np.ndarray, low: np.ndarray, window: int = 5
) -> Tuple[int, int]:
    """
    Find swing high and low points
    Args:
        high: Array of high prices
        low: Array of low prices
        window: Window size for finding swing points
    Returns:
        Tuple of (high_idx, low_idx)
    """
    high_idx = window
    low_idx = window
    # Find swing high
    for i in range(window, len(high) - window):
        if all(
            float(high[i]) > float(high[i - j]) for j in range(1, window + 1)
        ) and all(float(high[i]) > float(high[i + j]) for j in range(1, window + 1)):
            high_idx = i
            break
    # Find swing low
    for i in range(window, len(low) - window):
        if all(float(low[i]) < float(low[i - j]) for j in range(1, window + 1)) and all(
            float(low[i]) < float(low[i + j]) for j in range(1, window + 1)
        ):
            low_idx = i
            break
    return high_idx, low_idx


def calculate_fibonacci_levels(
    high: float, low: float, high_idx: int, low_idx: int
) -> FibonacciLevels:
    """
    Calculate Fibonacci retracement and extension levels
    Args:
        high: High price
        low: Low price
        high_idx: Index of high price
        low_idx: Index of low price
    Returns:
        FibonacciLevels object
    """
    # Calculate price range
    price_range = float(high - low)
    # Retracement levels
    retracement_levels = {
        0.0: float(high),
        0.236: float(high - 0.236 * price_range),
        0.382: float(high - 0.382 * price_range),
        0.5: float(high - 0.5 * price_range),
        0.618: float(high - 0.618 * price_range),
        0.786: float(high - 0.786 * price_range),
        1.0: float(low),
    }
    # Extension levels
    extension_levels = {
        1.272: float(low - 0.272 * price_range),
        1.618: float(low - 0.618 * price_range),
        2.0: float(low - price_range),
    }
    return FibonacciLevels(
        retracement=retracement_levels,
        extension=extension_levels,
        high=float(high),
        low=float(low),
        high_idx=high_idx,
        low_idx=low_idx,
    )


def get_fibonacci_levels(
    df: pd.DataFrame,
    from_idx: Optional[int] = None,
    to_idx: Optional[int] = None,
    window: int = 5,
) -> FibonacciLevels:
    """
    Get Fibonacci levels for a price series
    Args:
        df: DataFrame with OHLCV data
        from_idx: Start index for analysis
        to_idx: End index for analysis
        window: Window size for finding swing points
    Returns:
        FibonacciLevels object
    """
    # Get price data
    high = np.asarray(df["high"].values)
    low = np.asarray(df["low"].values)
    # Slice data if indices provided
    if from_idx is not None and to_idx is not None:
        high = high[from_idx : to_idx + 1]
        low = low[from_idx : to_idx + 1]
    # Find swing points
    high_idx, low_idx = find_swing_points(high, low, window)
    # Calculate levels
    return calculate_fibonacci_levels(
        float(high[high_idx]),
        float(low[low_idx]),
        high_idx,
        low_idx,
    )


def get_fibonacci_levels_manual(
    df: pd.DataFrame, high_idx: int, low_idx: int
) -> FibonacciLevels:
    """
    Get Fibonacci levels using manually specified high/low points
    Args:
        df: DataFrame with OHLCV data
        high_idx: Index of high price
        low_idx: Index of low price
    Returns:
        FibonacciLevels object
    """
    return calculate_fibonacci_levels(
        float(df["high"].iloc[high_idx]),
        float(df["low"].iloc[low_idx]),
        high_idx,
        low_idx,
    )


def plot_fibonacci_levels(
    fig: go.Figure, levels: FibonacciLevels, df: pd.DataFrame
) -> go.Figure:
    """
    Add Fibonacci levels to a Plotly figure
    Args:
        fig: Plotly figure object
        levels: FibonacciLevels object
        df: DataFrame with price data
    Returns:
        Updated Plotly figure
    """
    # Add retracement levels
    for level, price in levels.retracement.items():
        fig.add_hline(
            y=float(price),
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Fib {level:.3f}",
            annotation_position="right",
        )
    # Add extension levels
    for level, price in levels.extension.items():
        fig.add_hline(
            y=float(price),
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Ext {level:.3f}",
            annotation_position="right",
        )
    # Add high/low markers
    fig.add_trace(
        go.Scatter(
            x=[df.index[levels.high_idx]],
            y=[float(levels.high)],
            mode="markers",
            name="Swing High",
            marker=dict(color="red", size=10, symbol="triangle-down"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[df.index[levels.low_idx]],
            y=[float(levels.low)],
            mode="markers",
            name="Swing Low",
            marker=dict(color="green", size=10, symbol="triangle-up"),
        )
    )
    return fig
