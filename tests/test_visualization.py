import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from shared.visualization import (plot_cvd_and_delta_volume, plot_fuzzy_zones,
                                  plot_whale_activity)
@pytest.fixture
def price_data() -> Any:
    """Фикстура для ценовых данных"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "open": np.random.normal(50000, 1000, 100),
            "high": np.random.normal(51000, 1000, 100),
            "low": np.random.normal(49000, 1000, 100),
            "close": np.random.normal(50000, 1000, 100),
            "volume": np.random.normal(100, 20, 100),
        },
        index=dates,
    )
@pytest.fixture
def zones() -> Any:
    """Фикстура для зон поддержки/сопротивления"""
    return {
        "support": [(49000, 0.8), (48500, 0.6)],
        "resistance": [(51000, 0.7), (51500, 0.5)],
    }
@pytest.fixture
def trade_data() -> Any:
    """Фикстура для данных о сделках"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "open": np.random.normal(50000, 1000, 100),
            "high": np.random.normal(51000, 1000, 100),
            "low": np.random.normal(49000, 1000, 100),
            "close": np.random.normal(50000, 1000, 100),
            "volume": np.random.normal(100, 20, 100),
            "price": np.random.normal(50000, 1000, 100),
            "side": np.random.choice(["buy", "sell"], 100),
        },
        index=dates,
    )
    # Добавление нескольких крупных сделок
    df.loc[df.index[10], "volume"] = 500
    df.loc[df.index[20], "volume"] = 600
    df.loc[df.index[30], "volume"] = 700
    return df
@pytest.fixture
def volume_data() -> Any:
    """Фикстура для данных об объемах"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    cvd = pd.Series(np.cumsum(np.random.normal(0, 100, 100)), index=dates)
    delta = pd.Series(np.random.normal(0, 50, 100), index=dates)
    return cvd, delta
@pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    data = pd.DataFrame(
        {
            "open": np.random.normal(100, 1, 100),
            "high": np.random.normal(101, 1, 100),
            "low": np.random.normal(99, 1, 100),
            "close": np.random.normal(100, 1, 100),
            "volume": np.random.normal(1000, 100, 100),
        },
        index=dates,
    )
    return data
@pytest.fixture
def mock_trades() -> Any:
    """Фикстура с тестовыми сделками"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    return [
        {
            "entry_time": dates[0],
            "exit_time": dates[1],
            "entry_price": 100,
            "exit_price": 105,
            "size": 1,
            "pnl": 5,
        },
        {
            "entry_time": dates[2],
            "exit_time": dates[3],
            "entry_price": 105,
            "exit_price": 100,
            "size": 1,
            "pnl": -5,
        },
    ]
@pytest.fixture
def mock_indicators() -> Any:
    """Фикстура с тестовыми индикаторами"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    return {
        "rsi": pd.Series(np.random.uniform(0, 100, 100), index=dates),
        "macd": pd.Series(np.random.normal(0, 1, 100), index=dates),
        "signal": pd.Series(np.random.normal(0, 1, 100), index=dates),
        "hist": pd.Series(np.random.normal(0, 1, 100), index=dates),
    }
class TestVisualization:
    """Тесты для функций визуализации"""
    def test_plot_fuzzy_zones(self, price_data, zones) -> None:
        """Тест визуализации нечетких зон"""
        fig = plot_fuzzy_zones(price_data, zones)
        # Проверка наличия свечей
        assert len(fig.data) == 5  # 1 свечи + 4 зоны
        # Проверка типов данных
        assert isinstance(fig.data[0], go.Candlestick)
        assert all(isinstance(trace, go.Scatter) for trace in fig.data[1:])
        # Проверка layout
        assert fig.layout.title.text == "Fuzzy Support/Resistance Zones"
        assert fig.layout.yaxis.title.text == "Price"
        assert fig.layout.xaxis.title.text == "Time"
    def test_plot_whale_activity(self, trade_data) -> None:
        """Тест визуализации активности китов"""
        fig = plot_whale_activity(trade_data)
        # Проверка наличия свечей и маркеров
        assert len(fig.data) == 2  # свечи + маркеры китов
        # Проверка типов данных
        assert isinstance(fig.data[0], go.Candlestick)
        assert isinstance(fig.data[1], go.Scatter)
        # Проверка маркеров китов
        whale_trades = trade_data[
            trade_data["volume"] > trade_data["volume"].quantile(0.95)
        ]
        assert len(fig.data[1].x) == len(whale_trades)
        # Проверка layout
        assert fig.layout.title.text == "Whale Activity"
        assert fig.layout.yaxis.title.text == "Price"
        assert fig.layout.xaxis.title.text == "Time"
    def test_plot_cvd_and_delta_volume(self, price_data, volume_data) -> None:
        """Тест визуализации CVD и дельты объема"""
        cvd, delta = volume_data
        fig = plot_cvd_and_delta_volume(price_data, cvd, delta)
        # Проверка наличия всех компонентов
        assert len(fig.data) == 3  # свечи + CVD + дельта
        # Проверка типов данных
        assert isinstance(fig.data[0], go.Candlestick)
        assert isinstance(fig.data[1], go.Scatter)
        assert isinstance(fig.data[2], go.Bar)
        # Проверка layout
        assert fig.layout.title.text == "CVD and Volume Delta Analysis"
        assert fig.layout.yaxis.title.text == "Price"
        assert fig.layout.yaxis2.title.text == "CVD/Delta"
        assert fig.layout.xaxis.title.text == "Time"
    def test_error_handling(self) -> None:
        """Тест обработки ошибок"""
        # Тест с пустыми данными
        with pytest.raises(Exception):
            plot_fuzzy_zones(pd.DataFrame(), {})
        with pytest.raises(Exception):
            plot_whale_activity(pd.DataFrame())
        with pytest.raises(Exception):
            plot_cvd_and_delta_volume(pd.DataFrame(), pd.Series(), pd.Series())
        # Тест с некорректными данными
        with pytest.raises(Exception):
            plot_fuzzy_zones(None, None)
        with pytest.raises(Exception):
            plot_whale_activity(None)
        with pytest.raises(Exception):
            plot_cvd_and_delta_volume(None, None, None)
