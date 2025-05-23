from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from backtest.backtest import Backtest, Trade

from utils.logger import setup_logger

logger = setup_logger(__name__)


@pytest.fixture
def mock_market_data():
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
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
def backtest_config():
    """Фикстура с конфигурацией бэктеста"""
    return {
        "initial_balance": 10000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "position_size": 0.1,
        "stop_loss": 0.02,
        "take_profit": 0.05,
    }


@pytest.fixture
def backtest(backtest_config, mock_market_data):
    """Фикстура с экземпляром бэктеста"""
    return Backtest(data=mock_market_data, config=backtest_config)


class TestBacktest:
    def test_backtest_initialization(self, backtest, backtest_config):
        """Тест инициализации бэктеста"""
        assert backtest.initial_balance == backtest_config["initial_balance"]
        assert backtest.commission == backtest_config["commission"]
        assert backtest.slippage == backtest_config["slippage"]
        assert backtest.position_size == backtest_config["position_size"]
        assert backtest.stop_loss == backtest_config["stop_loss"]
        assert backtest.take_profit == backtest_config["take_profit"]
        assert isinstance(backtest.trades, list)
        assert isinstance(backtest.equity_curve, pd.Series)

    def test_run_backtest(self, backtest):
        """Тест запуска бэктеста"""
        strategy = Mock()
        strategy.generate_signal.return_value = {"action": "buy", "price": 100.0}

        results = backtest.run(strategy)
        assert isinstance(results, dict)
        assert "trades" in results
        assert "equity_curve" in results
        assert "metrics" in results

    def test_open_position(self, backtest):
        """Тест открытия позиции"""
        trade = backtest._open_position(
            price=100.0, size=0.1, direction="long", timestamp=datetime.now()
        )
        assert isinstance(trade, Trade)
        assert trade.entry_price == 100.0
        assert trade.size == 0.1
        assert trade.direction == "long"
        assert trade.status == "open"

    def test_close_position(self, backtest):
        """Тест закрытия позиции"""
        trade = backtest._open_position(
            price=100.0, size=0.1, direction="long", timestamp=datetime.now()
        )
        closed_trade = backtest._close_position(
            trade=trade, price=105.0, timestamp=datetime.now()
        )
        assert closed_trade.exit_price == 105.0
        assert closed_trade.status == "closed"
        assert closed_trade.pnl == 0.5  # (105 - 100) * 0.1

    def test_calculate_position_size(self, backtest):
        """Тест расчета размера позиции"""
        size = backtest._calculate_position_size(price=100.0, risk_per_trade=0.02)
        assert isinstance(size, float)
        assert size > 0
        assert size <= backtest.position_size

    def test_get_trade_statistics(self, backtest):
        """Тест получения статистики по сделкам"""
        # Добавляем тестовые сделки
        for i in range(5):
            trade = Trade(
                symbol="BTC/USDT",
                direction="long" if i % 2 == 0 else "short",
                entry_price=100.0,
                size=0.1,
                entry_time=datetime.now(),
                exit_price=105.0 if i % 2 == 0 else 95.0,
                exit_time=datetime.now() + timedelta(hours=1),
                pnl=0.5 if i % 2 == 0 else -0.5,
                status="closed",
            )
            backtest.trades.append(trade)

        stats = backtest.get_trade_statistics()
        assert isinstance(stats, dict)
        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "avg_profit" in stats
        assert "avg_loss" in stats
        assert "profit_factor" in stats
