from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    signals: Optional[Dict[str, List[int]]] = None,
    indicators: Optional[Dict[str, pd.Series]] = None,
    title: str = "Price Chart",
) -> go.Figure:
    """
    Create interactive candlestick chart with signals and indicators

    Args:
        df: DataFrame with OHLCV data
        signals: Dict of signal types and their indices
        indicators: Dict of indicator names and their values
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # Add volume bars
    colors = ["red" if row["close"] < row["open"] else "green" for _, row in df.iterrows()]

    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors), row=2, col=1
    )

    # Add signals if provided
    if signals:
        for signal_type, indices in signals.items():
            if signal_type == "buy":
                marker_color = "green"
                marker_symbol = "triangle-up"
            else:  # sell
                marker_color = "red"
                marker_symbol = "triangle-down"

            fig.add_trace(
                go.Scatter(
                    x=df.index[indices],
                    y=df["close"].iloc[indices],
                    mode="markers",
                    name=signal_type,
                    marker=dict(color=marker_color, size=10, symbol=marker_symbol),
                ),
                row=1,
                col=1,
            )

    # Add indicators if provided
    if indicators:
        for name, values in indicators.items():
            fig.add_trace(
                go.Scatter(x=df.index, y=values, name=name, line=dict(width=1)), row=1, col=1
            )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        height=800,
    )

    return fig


def plot_pnl_curves(returns: Dict[str, pd.Series], title: str = "Strategy Comparison") -> go.Figure:
    """
    Plot PnL curves for multiple strategies

    Args:
        returns: Dict of strategy names and their returns
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    for name, rets in returns.items():
        cumulative_returns = (1 + rets).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns, name=name, mode="lines")
        )

    fig.update_layout(title=title, yaxis_title="Cumulative Returns", xaxis_title="Date", height=600)

    return fig


def plot_volatility_drawdown(
    returns: pd.Series, window: int = 20, title: str = "Volatility and Drawdown"
) -> go.Figure:
    """
    Plot rolling volatility and drawdown

    Args:
        returns: Series of returns
        window: Rolling window size
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Calculate metrics
    volatility = returns.rolling(window).std() * np.sqrt(252)
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max

    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # Add volatility
    fig.add_trace(
        go.Scatter(x=volatility.index, y=volatility, name="Volatility", line=dict(color="blue")),
        row=1,
        col=1,
    )

    # Add drawdown
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", line=dict(color="red")),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title, yaxis_title="Annualized Volatility", yaxis2_title="Drawdown", height=800
    )

    return fig


def export_chart(
    fig: go.Figure, filename: str, format: str = "html", width: int = 1200, height: int = 800
) -> None:
    """
    Export chart to file

    Args:
        fig: Plotly figure object
        filename: Output filename
        format: Output format ('html' or 'png')
        width: Image width
        height: Image height
    """
    if format == "html":
        pio.write_html(fig, filename)
    elif format == "png":
        pio.write_image(fig, filename, width=width, height=height)
    else:
        raise ValueError(f"Unsupported format: {format}")


def plot_strategy_comparison(
    strategies: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    title: str = "Strategy Comparison",
) -> go.Figure:
    """
    Create comprehensive strategy comparison chart

    Args:
        strategies: Dict of strategy names and their returns
        benchmark: Optional benchmark returns
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25]
    )

    # Plot cumulative returns
    for name, returns in strategies.items():
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns, name=name, mode="lines"),
            row=1,
            col=1,
        )

    if benchmark is not None:
        benchmark_cumulative = (1 + benchmark).cumprod()
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative,
                name="Benchmark",
                line=dict(dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Plot rolling correlation with benchmark
    if benchmark is not None:
        for name, returns in strategies.items():
            correlation = returns.rolling(20).corr(benchmark)
            fig.add_trace(
                go.Scatter(
                    x=correlation.index, y=correlation, name=f"{name} Correlation", mode="lines"
                ),
                row=2,
                col=1,
            )

    # Plot rolling Sharpe ratio
    for name, returns in strategies.items():
        sharpe = returns.rolling(20).mean() / returns.rolling(20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=sharpe.index, y=sharpe, name=f"{name} Sharpe", mode="lines"), row=3, col=1
        )

    fig.update_layout(
        title=title,
        yaxis_title="Cumulative Returns",
        yaxis2_title="Correlation",
        yaxis3_title="Sharpe Ratio",
        height=1000,
    )

    return fig


def plot_fuzzy_zones(
    price_data: pd.DataFrame, zones: Dict[str, List[Tuple[float, float]]]
) -> go.Figure:
    """
    Визуализация нечетких зон поддержки и сопротивления.

    Args:
        price_data: DataFrame с ценами (OHLCV)
        zones: Словарь зон {type: [(price, strength), ...]}

    Returns:
        go.Figure: График с зонами
    """
    try:
        # Создание графика
        fig = go.Figure()

        # Добавление свечей
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data["open"],
                high=price_data["high"],
                low=price_data["low"],
                close=price_data["close"],
                name="Price",
            )
        )

        # Добавление зон
        colors = {"support": "rgba(0, 255, 0, 0.2)", "resistance": "rgba(255, 0, 0, 0.2)"}

        for zone_type, zone_list in zones.items():
            for price, strength in zone_list:
                fig.add_trace(
                    go.Scatter(
                        x=[price_data.index[0], price_data.index[-1]],
                        y=[price, price],
                        mode="lines",
                        line=dict(
                            color=colors.get(zone_type, "rgba(128, 128, 128, 0.2)"),
                            width=2,
                            dash="dash",
                        ),
                        name=f"{zone_type} ({strength:.2f})",
                    )
                )

        # Настройка layout
        fig.update_layout(
            title="Fuzzy Support/Resistance Zones",
            yaxis_title="Price",
            xaxis_title="Time",
            template="plotly_dark",
        )

        return fig

    except Exception as e:
        logger.error(f"Error plotting fuzzy zones: {str(e)}")
        raise


def plot_whale_activity(trade_data: pd.DataFrame) -> go.Figure:
    """
    Визуализация активности китов.

    Args:
        trade_data: DataFrame с данными о сделках

    Returns:
        go.Figure: График с активностью китов
    """
    try:
        # Создание графика
        fig = go.Figure()

        # Добавление свечей
        fig.add_trace(
            go.Candlestick(
                x=trade_data.index,
                open=trade_data["open"],
                high=trade_data["high"],
                low=trade_data["low"],
                close=trade_data["close"],
                name="Price",
            )
        )

        # Добавление крупных сделок
        whale_trades = trade_data[trade_data["volume"] > trade_data["volume"].quantile(0.95)]

        fig.add_trace(
            go.Scatter(
                x=whale_trades.index,
                y=whale_trades["price"],
                mode="markers",
                marker=dict(
                    size=whale_trades["volume"] / whale_trades["volume"].max() * 20,
                    color=whale_trades["side"].map({"buy": "green", "sell": "red"}),
                    symbol="diamond",
                ),
                name="Whale Trades",
            )
        )

        # Настройка layout
        fig.update_layout(
            title="Whale Activity", yaxis_title="Price", xaxis_title="Time", template="plotly_dark"
        )

        return fig

    except Exception as e:
        logger.error(f"Error plotting whale activity: {str(e)}")
        raise


def plot_cvd_and_delta_volume(
    price_data: pd.DataFrame, cvd: pd.Series, delta: pd.Series
) -> go.Figure:
    """
    Визуализация CVD и дельты объема.

    Args:
        price_data: DataFrame с ценами
        cvd: Series с CVD
        delta: Series с дельтой объема

    Returns:
        go.Figure: График с CVD и дельтой
    """
    try:
        # Создание графика с двумя подграфиками
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3]
        )

        # Добавление свечей
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data["open"],
                high=price_data["high"],
                low=price_data["low"],
                close=price_data["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Добавление CVD
        fig.add_trace(
            go.Scatter(x=cvd.index, y=cvd.values, name="CVD", line=dict(color="blue")), row=2, col=1
        )

        # Добавление дельты объема
        fig.add_trace(
            go.Bar(
                x=delta.index,
                y=delta.values,
                name="Volume Delta",
                marker_color="rgba(0, 255, 0, 0.5)",
            ),
            row=2,
            col=1,
        )

        # Настройка layout
        fig.update_layout(
            title="CVD and Volume Delta Analysis",
            yaxis_title="Price",
            yaxis2_title="CVD/Delta",
            xaxis_title="Time",
            template="plotly_dark",
        )

        return fig

    except Exception as e:
        logger.error(f"Error plotting CVD and delta: {str(e)}")
        raise


def plot_equity_curve(equity: pd.Series, trades: List[Dict[str, Any]] = None) -> go.Figure:
    """Построение графика эквити"""
    fig = go.Figure()

    # Добавляем линию эквити
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="blue"))
    )

    # Добавляем точки входа/выхода
    if trades:
        for trade in trades:
            # Точка входа
            fig.add_trace(
                go.Scatter(
                    x=[trade["entry_time"]],
                    y=[trade["entry_price"]],
                    mode="markers",
                    name="Entry",
                    marker=dict(color="green", size=10),
                )
            )

            # Точка выхода
            if "exit_time" in trade and "exit_price" in trade:
                fig.add_trace(
                    go.Scatter(
                        x=[trade["exit_time"]],
                        y=[trade["exit_price"]],
                        mode="markers",
                        name="Exit",
                        marker=dict(color="red", size=10),
                    )
                )

    fig.update_layout(
        title="Equity Curve", xaxis_title="Time", yaxis_title="Equity", showlegend=True
    )

    return fig


def plot_trades(data: pd.DataFrame, trades: List[Dict[str, Any]]) -> go.Figure:
    """Построение графика сделок"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Price", "Volume")
    )

    # График цены
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # График объема
    fig.add_trace(go.Bar(x=data.index, y=data["volume"], name="Volume"), row=2, col=1)

    # Добавляем сделки
    for trade in trades:
        # Точка входа
        fig.add_trace(
            go.Scatter(
                x=[trade["entry_time"]],
                y=[trade["entry_price"]],
                mode="markers",
                name="Entry",
                marker=dict(color="green", size=10),
            ),
            row=1,
            col=1,
        )

        # Точка выхода
        if "exit_time" in trade and "exit_price" in trade:
            fig.add_trace(
                go.Scatter(
                    x=[trade["exit_time"]],
                    y=[trade["exit_price"]],
                    mode="markers",
                    name="Exit",
                    marker=dict(color="red", size=10),
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        title="Trading Chart", xaxis_title="Time", yaxis_title="Price", showlegend=True
    )

    return fig


def plot_indicators(data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> go.Figure:
    """Построение графика индикаторов"""
    fig = make_subplots(rows=len(indicators) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # График цены
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # Добавляем индикаторы
    for i, (name, values) in enumerate(indicators.items(), start=2):
        fig.add_trace(go.Scatter(x=data.index, y=values, name=name), row=i, col=1)

    fig.update_layout(title="Technical Indicators", xaxis_title="Time", showlegend=True)

    return fig
