from datetime import datetime

import pytest

from core.controllers.risk_controller import RiskController
from core.models import Position


@pytest.fixture
def config():
    return {
        "balance": 10000.0,
        "stop_loss_multiplier": 2.0,
        "take_profit_multiplier": 3.0,
        "max_position_size": 1.0,
        "max_daily_loss": 1000.0,
    }


@pytest.fixture
def risk_controller(config):
    return RiskController(config)


@pytest.fixture
def sample_position():
    return Position(
        pair="BTC/USDT",
        side="long",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        entry_time=datetime.now(),
    )


def test_calculate_position_size(risk_controller):
    """Тест расчета размера позиции"""
    size = risk_controller.calculate_position_size("BTC/USDT", 50000.0, 0.02)
    assert size == 0.004  # 10000 * 0.02 / 50000


def test_calculate_stop_loss_long(risk_controller, sample_position):
    """Тест расчета стоп-лосса для длинной позиции"""
    stop_loss = risk_controller.calculate_stop_loss(sample_position, 1000.0)
    assert stop_loss == 48000.0  # 50000 - (1000 * 2)


def test_calculate_stop_loss_short(risk_controller):
    """Тест расчета стоп-лосса для короткой позиции"""
    position = Position(
        pair="BTC/USDT",
        side="short",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        entry_time=datetime.now(),
    )
    stop_loss = risk_controller.calculate_stop_loss(position, 1000.0)
    assert stop_loss == 52000.0  # 50000 + (1000 * 2)


def test_calculate_take_profit_long(risk_controller, sample_position):
    """Тест расчета тейк-профита для длинной позиции"""
    take_profit = risk_controller.calculate_take_profit(sample_position, 1000.0)
    assert take_profit == 53000.0  # 50000 + (1000 * 3)


def test_calculate_take_profit_short(risk_controller):
    """Тест расчета тейк-профита для короткой позиции"""
    position = Position(
        pair="BTC/USDT",
        side="short",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        entry_time=datetime.now(),
    )
    take_profit = risk_controller.calculate_take_profit(position, 1000.0)
    assert take_profit == 47000.0  # 50000 - (1000 * 3)


def test_check_risk_limits_valid(risk_controller, sample_position):
    """Тест проверки лимитов риска - валидный случай"""
    risk_controller.risk_metrics["daily_pnl"] = 0.0
    assert risk_controller.check_risk_limits(sample_position) is True


def test_check_risk_limits_invalid_size(risk_controller):
    """Тест проверки лимитов риска - превышение размера позиции"""
    position = Position(
        pair="BTC/USDT",
        side="long",
        size=2.0,  # Превышает max_position_size
        entry_price=50000.0,
        current_price=50000.0,
        pnl=0.0,
        entry_time=datetime.now(),
    )
    assert risk_controller.check_risk_limits(position) is False


def test_check_risk_limits_invalid_daily_loss(risk_controller, sample_position):
    """Тест проверки лимитов риска - превышение дневного убытка"""
    risk_controller.risk_metrics["daily_pnl"] = -2000.0  # Превышает max_daily_loss
    assert risk_controller.check_risk_limits(sample_position) is False


def test_update_risk_metrics(risk_controller):
    """Тест обновления метрик риска"""
    positions = [
        Position(
            pair="BTC/USDT",
            side="long",
            size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            pnl=100.0,
            entry_time=datetime.now(),
        ),
        Position(
            pair="ETH/USDT",
            side="short",
            size=1.0,
            entry_price=3000.0,
            current_price=2900.0,
            pnl=100.0,
            entry_time=datetime.now(),
        ),
    ]

    risk_controller.update_risk_metrics(positions)

    assert risk_controller.risk_metrics["total_pnl"] == 200.0
    assert risk_controller.risk_metrics["win_rate"] == 1.0
    assert risk_controller.risk_metrics["position_count"] == 2
