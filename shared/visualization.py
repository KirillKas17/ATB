# -*- coding: utf-8 -*-
"""Модуль визуализации для shared слоя."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from infrastructure.core.visualization import plot_whale_activity, plot_trades


def plot_equity_curve(
    equity_data: pd.Series, title: str = "Equity Curve"
) -> plt.Figure:
    """Построение графика кривой доходности."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_data.index, equity_data.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True)
    return fig


def plot_indicators(
    prices: pd.Series,
    indicators: Dict[str, pd.Series],
    title: str = "Technical Indicators",
) -> plt.Figure:
    """Построение графика с техническими индикаторами."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # График цен
    ax1.plot(prices.index, prices.to_numpy(), label="Price")
    ax1.set_title(title)
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)
    # График индикаторов
    for name, indicator in indicators.items():
        ax2.plot(indicator.index, indicator.to_numpy(), label=name)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Indicator Value")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    return fig


def plot_cvd_and_delta_volume(
    cvd: pd.Series, delta_volume: pd.Series, title: str = "CVD and Delta Volume"
) -> plt.Figure:
    """Построение графика CVD и дельта объема."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # График CVD
    ax1.plot(cvd.index, cvd.to_numpy(), label="CVD", color="blue")
    ax1.set_title(title)
    ax1.set_ylabel("CVD")
    ax1.legend()
    ax1.grid(True)
    # График дельта объема
    ax2.plot(delta_volume.index, delta_volume.to_numpy(), label="Delta Volume", color="red")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Delta Volume")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    return fig


def plot_fuzzy_zones(
    zones: List[Dict[str, Any]], prices: pd.Series, title: str = "Fuzzy Zones"
) -> plt.Figure:
    """Построение графика нечетких зон."""
    fig, ax = plt.subplots(figsize=(12, 6))
    # График цен
    ax.plot(prices.index, prices.to_numpy(), label="Price", color="black")
    # Отображение зон
    colors = ["red", "green", "blue", "yellow", "purple"]
    for i, zone in enumerate(zones):
        color = colors[i % len(colors)]
        ax.axhspan(
            zone["low"], zone["high"], alpha=0.3, color=color, label=f"Zone {i+1}"
        )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig


def plot_volume_profile(
    volume_profile: Dict[str, Any], title: str = "Volume Profile"
) -> plt.Figure:
    """Построение графика профиля объема."""
    fig, ax = plt.subplots(figsize=(10, 8))
    prices = volume_profile.get("prices", [])
    volumes = volume_profile.get("volumes", [])
    ax.barh(prices, volumes, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.grid(True)
    return fig


def plot_order_book(
    order_book: Dict[str, List], title: str = "Order Book"
) -> plt.Figure:
    """Построение графика ордербука."""
    fig, ax = plt.subplots(figsize=(12, 6))
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if bids:
        bid_prices, bid_volumes = zip(*bids)
        ax.barh(bid_prices, bid_volumes, color="green", alpha=0.7, label="Bids")
    if asks:
        ask_prices, ask_volumes = zip(*asks)
        ax.barh(ask_prices, ask_volumes, color="red", alpha=0.7, label="Asks")
    ax.set_title(title)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig


def plot_correlation_matrix(
    correlation_matrix: np.ndarray, labels: List[str], title: str = "Correlation Matrix"
) -> plt.Figure:
    """Построение матрицы корреляций."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    # Добавление подписей
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    # Добавление цветовой шкалы
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return fig


def plot_heatmap(
    data: np.ndarray, x_labels: List[str], y_labels: List[str], title: str = "Heatmap"
) -> plt.Figure:
    """Построение тепловой карты."""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap="viridis")
    # Добавление подписей
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_yticklabels(y_labels)
    # Добавление цветовой шкалы
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return fig


def plot_distribution(data: pd.Series, title: str = "Distribution") -> plt.Figure:
    """Построение графика распределения."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data.to_numpy(), bins=50, alpha=0.7, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    return fig


def plot_time_series(data: pd.Series, title: str = "Time Series") -> plt.Figure:
    """Построение временного ряда."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    return fig
