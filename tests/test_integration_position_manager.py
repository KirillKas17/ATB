from datetime import datetime, timedelta
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from core.position_manager import (Position, PositionManager, PositionSide,
                                   PositionStatus)
from shared.logging import setup_logger
logger = setup_logger(__name__)
    @pytest.fixture
def mock_market_data() -> Any:
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
def position_manager() -> Any:
    """Фикстура с менеджером позиций"""
    return PositionManager(
        config={
            "max_positions": 10,
            "max_position_size": 1.0,
            "max_drawdown": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "trailing_stop": 0.01,
            "position_timeout": 24,
        }
    )
    @pytest.fixture
def mock_position() -> Any:
    """Фикстура с тестовой позицией"""
    return Position(
        symbol="BTC/USDT",
        side="buy",
        entry_price=100.0,
        size=0.1,
        stop_loss=98.0,
        take_profit=105.0,
        entry_time=datetime.now(),
        status="open",
    )
class TestPosition:
    def test_position_initialization(self, mock_position) -> None:
        """Тест инициализации позиции"""
        assert mock_position.symbol == "BTC/USDT"
        assert mock_position.side == PositionSide.LONG
        assert mock_position.size == 1.0
        assert mock_position.entry_price == 50000.0
        assert mock_position.stop_loss == 49000.0
        assert mock_position.take_profit == 51000.0
        assert mock_position.status == PositionStatus.OPEN
        assert mock_position.leverage == 1.0
        assert mock_position.fees == 0.0
        assert mock_position.tags == []
        assert mock_position.metadata == {}
    def test_calculate_margin(self, mock_position) -> None:
        """Тест расчета маржи"""
        margin = mock_position.calculate_margin()
        assert margin == 50000.0  # size * entry_price / leverage
    def test_calculate_pnl_long(self, mock_position) -> None:
        """Тест расчета P&L для длинной позиции"""
        pnl = mock_position.calculate_pnl(51000.0)
        assert pnl == 1000.0  # (51000 - 50000) * 1.0
    def test_calculate_pnl_short(self: "TestPosition") -> None:
        """Тест расчета P&L для короткой позиции"""
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            size=1.0,
            entry_price=50000.0,
            stop_loss=51000.0,
            take_profit=49000.0,
            entry_time=datetime.now(),
        )
        pnl = position.calculate_pnl(49000.0)
        assert pnl == 1000.0  # (50000 - 49000) * 1.0
    def test_calculate_roi(self, mock_position) -> None:
        """Тест расчета ROI"""
        mock_position.pnl = 1000.0
        roi = mock_position.calculate_roi()
        assert roi == 2.0  # (1000 / 50000) * 100
    def test_is_profitable(self, mock_position) -> None:
        """Тест проверки прибыльности"""
        mock_position.pnl = 1000.0
        assert mock_position.is_profitable()
        mock_position.pnl = -1000.0
        assert not mock_position.is_profitable()
    def test_get_duration(self, mock_position) -> None:
        """Тест расчета длительности позиции"""
        mock_position.exit_time = mock_position.entry_time + timedelta(hours=1)
        duration = mock_position.get_duration()
        assert duration == 1.0
class TestPositionManager:
    def test_open_position(self, position_manager, mock_position) -> None:
        """Тест открытия позиции"""
        position = position_manager.open_position(
            symbol="BTC/USDT",
            side="buy",
            entry_price=100.0,
            size=0.1,
            stop_loss=98.0,
            take_profit=105.0,
        )
        assert position is not None
        assert position.symbol == "BTC/USDT"
        assert position.side == "buy"
        assert position.entry_price == 100.0
        assert position.size == 0.1
        assert position.stop_loss == 98.0
        assert position.take_profit == 105.0
        assert position.status == "open"
    def test_close_position(self, position_manager, mock_position) -> None:
        """Тест закрытия позиции"""
        position_manager.positions.append(mock_position)
        result = position_manager.close_position(
            symbol="BTC/USDT", exit_price=102.0, exit_time=datetime.now()
        )
        assert result is True
        assert mock_position.status == "closed"
        assert mock_position.exit_price == 102.0
        assert mock_position.exit_time is not None
    def test_update_position(self, position_manager, mock_position) -> None:
        """Тест обновления позиции"""
        position_manager.positions.append(mock_position)
        current_price = 101.0
        position_manager.update_position(mock_position, current_price)
        assert mock_position.current_price == current_price
        assert mock_position.pnl is not None
    def test_get_win_rate(self, position_manager) -> None:
        """Тест получения винрейта"""
        # Добавляем тестовые позиции
        for i in range(10):
            position = Position(
                symbol="BTC/USDT",
                side="buy",
                entry_price=100.0,
                size=0.1,
                stop_loss=98.0,
                take_profit=105.0,
                entry_time=datetime.now(),
                status="closed",
                exit_price=105.0 if i < 7 else 97.0,  # 7 выигрышных, 3 проигрышных
                exit_time=datetime.now(),
            )
            position_manager.positions.append(position)
        win_rate = position_manager.get_win_rate()
        assert win_rate == 0.7  # 7/10 = 0.7
    def test_get_average_roi(self, position_manager) -> None:
        """Тест получения среднего ROI"""
        # Добавляем тестовые позиции
        for i in range(5):
            position = Position(
                symbol="BTC/USDT",
                side="buy",
                entry_price=100.0,
                size=0.1,
                stop_loss=98.0,
                take_profit=105.0,
                entry_time=datetime.now(),
                status="closed",
                exit_price=105.0,  # 5% прибыль
                exit_time=datetime.now(),
            )
            position_manager.positions.append(position)
        avg_roi = position_manager.get_average_roi()
        assert avg_roi == 0.05  # 5% ROI
    def test_get_max_drawdown(self, position_manager) -> None:
        """Тест получения максимальной просадки"""
        # Добавляем тестовые позиции с просадкой
        position = Position(
            symbol="BTC/USDT",
            side="buy",
            entry_price=100.0,
            size=0.1,
            stop_loss=98.0,
            take_profit=105.0,
            entry_time=datetime.now(),
            status="closed",
            exit_price=95.0,  # 5% просадка
            exit_time=datetime.now(),
        )
        position_manager.positions.append(position)
        max_drawdown = position_manager.get_max_drawdown()
        assert max_drawdown == 0.05  # 5% просадка
    def test_get_position_correlation(self, position_manager) -> None:
        """Тест получения корреляции позиций"""
        # Добавляем тестовые позиции
        for i in range(5):
            position = Position(
                symbol="BTC/USDT",
                side="buy",
                entry_price=100.0,
                size=0.1,
                stop_loss=98.0,
                take_profit=105.0,
                entry_time=datetime.now(),
                status="closed",
                exit_price=105.0,
                exit_time=datetime.now(),
            )
            position_manager.positions.append(position)
        correlation = position_manager.get_position_correlation()
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1
    def test_risk_metrics(self, position_manager) -> None:
        """Тест расчета метрик риска"""
        # Добавляем тестовые позиции
        for i in range(10):
            position = Position(
                symbol="BTC/USDT",
                side="buy",
                entry_price=100.0,
                size=0.1,
                stop_loss=98.0,
                take_profit=105.0,
                entry_time=datetime.now(),
                status="closed",
                exit_price=105.0 if i < 7 else 97.0,
                exit_time=datetime.now(),
            )
            position_manager.positions.append(position)
        metrics = position_manager.calculate_risk_metrics()
        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "var_95" in metrics
        assert "max_drawdown" in metrics
