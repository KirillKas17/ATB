from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.data_loader import load_market_data, save_market_data
from utils.indicators import calculate_bollinger_bands, calculate_macd, calculate_rsi
from utils.logging import log_error, log_trade, setup_logger
from utils.math_utils import (
    calculate_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)
from utils.visualization import plot_equity_curve, plot_indicators, plot_trades


# Фикстуры
@pytest.fixture
def mock_market_data():
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
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
def mock_trades():
    """Фикстура с тестовыми сделками"""
    return [
        {"entry_price": 100, "exit_price": 105, "size": 1, "pnl": 5},
        {"entry_price": 105, "exit_price": 100, "size": 1, "pnl": -5},
    ]


# Тесты для индикаторов
class TestIndicators:
    def test_calculate_rsi(self, mock_market_data):
        """Тест расчета RSI"""
        rsi = calculate_rsi(mock_market_data["close"], period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(mock_market_data)
        assert all(0 <= x <= 100 for x in rsi.dropna())

    def test_calculate_macd(self, mock_market_data):
        """Тест расчета MACD"""
        macd, signal, hist = calculate_macd(
            mock_market_data["close"], fast_period=12, slow_period=26, signal_period=9
        )

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(mock_market_data)

    def test_calculate_bollinger_bands(self, mock_market_data):
        """Тест расчета полос Боллинджера"""
        upper, middle, lower = calculate_bollinger_bands(
            mock_market_data["close"], period=20, std_dev=2
        )

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(mock_market_data)
        assert all(upper >= middle)
        assert all(middle >= lower)


# Тесты для математических утилит
class TestMathUtils:
    def test_calculate_sharpe_ratio(self, mock_trades):
        """Тест расчета коэффициента Шарпа"""
        returns = [trade["pnl"] for trade in mock_trades]
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)

    def test_calculate_drawdown(self, mock_trades):
        """Тест расчета просадки"""
        equity_curve = pd.Series([10000, 10050, 10000])
        drawdown = calculate_drawdown(equity_curve)

        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(equity_curve)
        assert all(x <= 0 for x in drawdown)

    def test_calculate_win_rate(self, mock_trades):
        """Тест расчета процента выигрышных сделок"""
        win_rate = calculate_win_rate(mock_trades)

        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1


# Тесты для логирования
class TestLogging:
    def test_setup_logger(self):
        """Тест настройки логгера"""
        logger = setup_logger("test_logger", "test.log")

        assert logger is not None
        assert logger.name == "test_logger"

    def test_log_trade(self):
        """Тест логирования сделки"""
        trade = {"symbol": "BTC/USDT", "side": "buy", "price": 50000, "size": 0.1}

        with patch("logging.Logger.info") as mock_info:
            log_trade(trade)
            mock_info.assert_called_once()

    def test_log_error(self):
        """Тест логирования ошибки"""
        error = Exception("Test error")

        with patch("logging.Logger.error") as mock_error:
            log_error(error)
            mock_error.assert_called_once()


# Тесты для загрузки данных
class TestDataLoader:
    def test_load_market_data(self):
        """Тест загрузки рыночных данных"""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"close": [100, 101, 102]})
            data = load_market_data("test.csv")

            assert isinstance(data, pd.DataFrame)
            mock_read_csv.assert_called_once_with("test.csv")

    def test_save_market_data(self, mock_market_data):
        """Тест сохранения рыночных данных"""
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            save_market_data(mock_market_data, "test.csv")
            mock_to_csv.assert_called_once_with("test.csv")


# Тесты для визуализации
class TestVisualization:
    def test_plot_equity_curve(self, mock_market_data):
        """Тест построения графика эквити"""
        equity_curve = pd.Series([10000, 10050, 10000], index=mock_market_data.index)

        with patch("plotly.graph_objects.Figure.show") as mock_show:
            plot_equity_curve(equity_curve)
            mock_show.assert_called_once()

    def test_plot_trades(self, mock_market_data, mock_trades):
        """Тест построения графика сделок"""
        with patch("plotly.graph_objects.Figure.show") as mock_show:
            plot_trades(mock_market_data, mock_trades)
            mock_show.assert_called_once()

    def test_plot_indicators(self, mock_market_data):
        """Тест построения графика индикаторов"""
        indicators = {
            "rsi": calculate_rsi(mock_market_data["close"]),
            "macd": calculate_macd(mock_market_data["close"])[0],
        }

        with patch("plotly.graph_objects.Figure.show") as mock_show:
            plot_indicators(mock_market_data, indicators)
            mock_show.assert_called_once()
