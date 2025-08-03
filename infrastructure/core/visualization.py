"""
Модуль визуализации для торговых данных.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Добавляем типы для pandas
from pandas.core.series import Series as PandasSeries
from pandas.core.frame import DataFrame as PandasDataFrame

def create_candlestick_chart(
    df: DataFrame,
    signals: Optional[Dict[str, List[int]]] = None,
    indicators: Optional[Dict[str, Series]] = None,
    title: str = "Price Chart",
) -> go.Figure:
    """
    Создание свечного графика с сигналами и индикаторами.
    Args:
        df: DataFrame с OHLCV данными
        signals: Словарь с сигналами
        indicators: Словарь с индикаторами
        title: Заголовок графика
    Returns:
        Plotly figure объект
    """
    fig = go.Figure()
    
    # Основной свечной график
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )
    
    # Добавляем индикаторы
    if indicators:
        for name, indicator in indicators.items():
            if isinstance(indicator, Series) and not indicator.empty:
                fig.add_trace(
                    go.Scatter(
                        x=indicator.index,
                        y=indicator,
                        name=name,
                        mode="lines",
                    )
                )
    
    # Добавляем сигналы
    if signals:
        for signal_type, signal_indices in signals.items():
            if signal_indices:
                # Преобразуем список индексов в numpy array для совместимости с pandas
                signal_indices_array = np.array(signal_indices)
                signal_prices = df["close"].iloc[signal_indices_array]
                fig.add_trace(
                    go.Scatter(
                        x=signal_prices.index,
                        y=signal_prices,
                        mode="markers",
                        name=signal_type,
                        marker=dict(
                            size=10,
                            symbol="triangle-up" if "buy" in signal_type.lower() else "triangle-down",
                            color="green" if "buy" in signal_type.lower() else "red",
                        ),
                    )
                )
    
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        xaxis_title="Date",
        height=600,
    )
    
    return fig


def plot_pnl_curves(
    returns: Dict[str, Series], title: str = "Strategy Comparison"
) -> go.Figure:
    """
    Построение кривых P&L для стратегий.
    Args:
        returns: Словарь с доходностями стратегий
        title: Заголовок графика
    Returns:
        Plotly figure объект
    """
    fig = go.Figure()
    
    for name, rets in returns.items():
        if isinstance(rets, pd.Series) and not rets.empty:
            # Расчет кумулятивных доходностей
            try:
                cumulative_returns = (1 + rets).cumprod()
            except AttributeError:
                cumulative_returns = rets
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    name=name,
                    mode="lines",
                )
            )
    fig.update_layout(
        title=title, yaxis_title="Cumulative Returns", xaxis_title="Date", height=600
    )
    return fig


def plot_volatility_drawdown(
    returns: Series, window: int = 20, title: str = "Volatility and Drawdown"
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
    try:
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
    except AttributeError:
        volatility = returns
    try:
        cumulative_returns = (1 + returns).cumprod()
    except AttributeError:
        cumulative_returns = returns
    
    try:
        running_max = cumulative_returns.expanding().max()
    except AttributeError:
        running_max = cumulative_returns
    drawdown = (cumulative_returns - running_max) / running_max
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    # Add volatility
    fig.add_trace(
        go.Scatter(
            x=volatility.index, y=volatility, name="Volatility", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown, name="Drawdown", line=dict(color="red")
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        yaxis_title="Annualized Volatility",
        yaxis2_title="Drawdown",
        height=800,
    )
    return fig


def export_chart(
    fig: go.Figure,
    filename: str,
    format: str = "html",
    width: int = 1200,
    height: int = 800,
) -> None:
    """
    Export chart to file
    Args:
        fig: Plotly figure object
        filename: Output filename
        format: Output format (html, png, svg, pdf)
        width: Chart width
        height: Chart height
    """
    fig.update_layout(width=width, height=height)
    if format == "html":
        fig.write_html(filename)
    elif format == "png":
        fig.write_image(filename)
    elif format == "svg":
        fig.write_image(filename)
    elif format == "pdf":
        fig.write_image(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def plot_strategy_comparison(
    strategies: Dict[str, Series],
    benchmark: Optional[Series] = None,
    title: str = "Strategy Comparison",
) -> go.Figure:
    """
    Plot strategy comparison with benchmark
    Args:
        strategies: Dict of strategy names and their returns
        benchmark: Benchmark returns series
        title: Chart title
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    # Add benchmark if provided
    if benchmark is not None:
        try:
            cumulative_benchmark = (1 + benchmark).cumprod()
        except AttributeError:
            cumulative_benchmark = benchmark
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_benchmark.index,
                y=cumulative_benchmark,
                name="Benchmark",
                line=dict(color="black", dash="dash"),
            )
        )
    # Add strategies
    for name, rets in strategies.items():
        try:
            cumulative_returns = (1 + rets).cumprod()
        except AttributeError:
            cumulative_returns = rets
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                name=name,
                mode="lines",
            )
        )
    fig.update_layout(
        title=title,
        yaxis_title="Cumulative Returns",
        xaxis_title="Date",
        height=600,
        showlegend=True,
    )
    return fig


def plot_fuzzy_zones(
    price_data: DataFrame, zones: Dict[str, List[Tuple[float, float]]]
) -> go.Figure:
    """
    Plot fuzzy support and resistance zones
    Args:
        price_data: DataFrame with OHLCV data
        zones: Dict with support and resistance zones
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    # Add candlestick chart
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
    # Add support zones
    if "support" in zones:
        for zone in zones["support"]:
            fig.add_hline(
                y=zone[0],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support: {zone[0]:.2f}",
            )
    # Add resistance zones
    if "resistance" in zones:
        for zone in zones["resistance"]:
            fig.add_hline(
                y=zone[0],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance: {zone[0]:.2f}",
            )
    fig.update_layout(
        title="Price Chart with Support/Resistance Zones",
        yaxis_title="Price",
        xaxis_title="Date",
        height=600,
    )
    return fig


def plot_whale_activity(trade_data: DataFrame) -> go.Figure:
    """
    Построение графика активности китов.
    Args:
        trade_data: DataFrame с данными о сделках
    Returns:
        Plotly figure объект
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Large Trades", "Trade Volume Distribution")
    )
    
    # Фильтруем крупные сделки (киты)
    whale_threshold = trade_data["volume"].quantile(0.95)
    whale_trades = trade_data[trade_data["volume"] > whale_threshold]
    
    # График крупных сделок
    fig.add_trace(
        go.Scatter(
            x=whale_trades.index,
            y=whale_trades["price"],
            mode="markers",
            name="Whale Trades",
            marker=dict(
                size=whale_trades["volume"] / whale_trades["volume"].max() * 20,
                color=whale_trades["volume"],
                colorscale="Viridis",
                showscale=True,
            ),
        ),
        row=1, col=1
    )
    
    # Распределение объемов
    fig.add_trace(
        go.Histogram(
            x=trade_data["volume"],
            name="Volume Distribution",
            nbinsx=50,
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Whale Activity Analysis",
        height=800,
    )
    
    return fig


def plot_cvd_and_delta_volume(
    price_data: DataFrame, cvd: Series, delta: Series
) -> go.Figure:
    """
    Построение графика CVD и дельта-объема.
    Args:
        price_data: DataFrame с ценовыми данными
        cvd: Series с CVD
        delta: Series с дельта-объемом
    Returns:
        Plotly figure объект
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price", "Cumulative Volume Delta", "Delta Volume")
    )
    
    # Ценовой график
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data["open"],
            high=price_data["high"],
            low=price_data["low"],
            close=price_data["close"],
            name="Price",
        ),
        row=1, col=1
    )
    
    # CVD
    if isinstance(cvd, Series) and not cvd.empty:
        fig.add_trace(
            go.Scatter(
                x=cvd.index,
                y=cvd,
                name="CVD",
                line=dict(color="blue"),
            ),
            row=2, col=1
        )
    
    # Дельта-объем
    if isinstance(delta, Series) and not delta.empty:
        fig.add_trace(
            go.Bar(
                x=delta.index,
                y=delta,
                name="Delta Volume",
                marker_color=delta.gt(0).map({True: "green", False: "red"}) if isinstance(delta, (pd.Series, PandasSeries)) and hasattr(delta, "gt") else (["green" if x > 0 else "red" for x in delta] if hasattr(delta, "__iter__") and not isinstance(delta, pd.Series) else ["green"]),
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        title="CVD and Delta Volume Analysis",
        height=900,
    )
    
    return fig


def plot_equity_curve(
    equity: Series, trades: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:
    """
    Построение кривой доходности с отметками сделок.
    Args:
        equity: Series с кривой доходности
        trades: Список сделок
    Returns:
        Plotly figure объект
    """
    fig = go.Figure()
    
    # Основная кривая доходности
    if isinstance(equity, Series) and not equity.empty:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                name="Equity Curve",
                line=dict(color="blue"),
            )
        )
    
    # Отметки сделок
    if trades:
        for trade in trades:
            if "entry_time" in trade and "exit_time" in trade:
                # Вход в позицию
                fig.add_trace(
                    go.Scatter(
                        x=[trade["entry_time"]],
                        y=[trade.get("entry_price", 0)],
                        mode="markers",
                        name="Entry",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color="green",
                        ),
                        showlegend=False,
                    )
                )
                
                # Выход из позиции
                fig.add_trace(
                    go.Scatter(
                        x=[trade["exit_time"]],
                        y=[trade.get("exit_price", 0)],
                        mode="markers",
                        name="Exit",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color="red",
                        ),
                        showlegend=False,
                    )
                )
    
    fig.update_layout(
        title="Equity Curve",
        yaxis_title="Equity",
        xaxis_title="Date",
        height=600,
    )
    
    return fig


def plot_trades(data: DataFrame, trades: List[Dict[str, Any]]) -> go.Figure:
    """
    Построение графика с отметками сделок.
    Args:
        data: DataFrame с ценовыми данными
        trades: Список сделок
    Returns:
        Plotly figure объект
    """
    fig = go.Figure()
    
    # Основной ценовой график
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
        )
    )
    
    # Отметки сделок
    for trade in trades:
        if "entry_time" in trade and "exit_time" in trade:
            # Вход в позицию
            fig.add_trace(
                go.Scatter(
                    x=[trade["entry_time"]],
                    y=[trade.get("entry_price", 0)],
                    mode="markers",
                    name="Entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color="green",
                    ),
                    showlegend=False,
                )
            )
            
            # Выход из позиции
            fig.add_trace(
                go.Scatter(
                    x=[trade["exit_time"]],
                    y=[trade.get("exit_price", 0)],
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="red",
                    ),
                    showlegend=False,
                )
            )
    
    fig.update_layout(
        title="Trades on Price Chart",
        yaxis_title="Price",
        xaxis_title="Date",
        height=600,
    )
    
    return fig


def plot_indicators(data: DataFrame, indicators: Dict[str, Series]) -> go.Figure:
    """
    Построение графика с техническими индикаторами.
    Args:
        data: DataFrame с ценовыми данными
        indicators: Словарь с индикаторами
    Returns:
        Plotly figure объект
    """
    fig = make_subplots(
        rows=len(indicators) + 1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Price"] + list(indicators.keys())
    )
    
    # Основной ценовой график
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
        ),
        row=1, col=1
    )
    
    # Индикаторы
    for i, (name, indicator) in enumerate(indicators.items(), 2):
        if isinstance(indicator, Series) and not indicator.empty:
            fig.add_trace(
                go.Scatter(
                    x=indicator.index,
                    y=indicator,
                    name=name,
                    line=dict(color="blue"),
                ),
                row=i, col=1
            )
    
    fig.update_layout(
        title="Technical Indicators",
        height=200 * (len(indicators) + 1),
    )
    
    return fig


# Основной класс для удобства использования
class Visualizer:
    """Главный класс визуализации для удобного доступа ко всем функциям."""
    
    def __init__(self):
        self.theme = "plotly_dark"
        self.default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    def create_price_chart(self, data, title="Price Chart"):
        """Создание графика цен."""
        return create_price_chart(data, title)
    
    def create_candlestick_chart(self, data, title="Candlestick Chart"):
        """Создание свечного графика."""
        return create_candlestick_chart(data, title)
    
    def create_volume_chart(self, data, title="Volume Chart"):
        """Создание графика объёмов."""
        return create_volume_chart(data, title)
    
    def create_portfolio_chart(self, portfolio_data, title="Portfolio"):
        """Создание графика портфеля."""
        return create_portfolio_chart(portfolio_data, title)
    
    def create_performance_chart(self, performance_data, title="Performance"):
        """Создание графика производительности."""
        return create_performance_chart(performance_data, title)
    
    def create_correlation_matrix(self, correlation_data, title="Correlation Matrix"):
        """Создание матрицы корреляций."""
        return create_correlation_matrix(correlation_data, title)
    
    def create_technical_indicators_chart(self, data, indicators, title="Technical Indicators"):
        """Создание графика технических индикаторов."""
        return create_technical_indicators_chart(data, indicators, title)
