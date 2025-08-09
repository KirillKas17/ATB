from datetime import datetime, timedelta

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.core.market_state import MarketState, MarketStateManager


@pytest.fixture
def sample_market_state() -> Any:
    """Фикстура с тестовым состоянием рынка"""
    return MarketState(
        timestamp=datetime.now(),
        price=100.0,
        volume=1000.0,
        volatility=0.02,
        trend="up",
        indicators={"rsi": 65.0, "macd": 0.5},
        market_regime="bull",
        liquidity=1000000.0,
        momentum=0.05,
        sentiment=0.7,
        support_levels=[95.0, 90.0],
        resistance_levels=[105.0, 110.0],
        market_depth={"bid": 500.0, "ask": 500.0},
        correlation_matrix={"BTC/USD": {"ETH/USD": 0.8}},
        market_impact=0.001,
        volume_profile={100.0: 1000.0, 101.0: 800.0},
    )

@pytest.fixture
def market_state_manager() -> Any:
    """Фикстура с менеджером состояния рынка"""
    config = {
        "lookback_period": 5,
        "regime_threshold": 0.7,
        "volatility_window": 3,
        "momentum_window": 3,
    }
    return MarketStateManager(config)


def test_market_state_creation(sample_market_state) -> None:
    """Тест создания состояния рынка"""
    assert sample_market_state.price == 100.0
    assert sample_market_state.volume == 1000.0
    assert sample_market_state.market_regime == "bull"
    assert len(sample_market_state.support_levels) == 2
    assert len(sample_market_state.resistance_levels) == 2

def test_market_state_serialization(sample_market_state) -> None:
    """Тест сериализации/десериализации состояния рынка"""
    state_dict = sample_market_state.to_dict()
    new_state = MarketState.from_dict(state_dict)

    assert new_state.price == sample_market_state.price
    assert new_state.volume == sample_market_state.volume
    assert new_state.market_regime == sample_market_state.market_regime
    assert new_state.support_levels == sample_market_state.support_levels
    assert new_state.resistance_levels == sample_market_state.resistance_levels


def test_market_state_manager_add_state(market_state_manager, sample_market_state) -> None:
    """Тест добавления состояния в менеджер"""
    market_state_manager.add_state(sample_market_state)
    latest_state = market_state_manager.get_latest_state()

    assert latest_state is not None
    assert latest_state.price == sample_market_state.price
    assert latest_state.volume == sample_market_state.volume


def test_market_state_manager_lookback(market_state_manager) -> None:
    """Тест ограничения истории состояний"""
    # Создаем и добавляем больше состояний, чем lookback_period
    for i in range(10):
        state = MarketState(
            timestamp=datetime.now() + timedelta(minutes=i),
            price=100.0 + i,
            volume=1000.0,
            volatility=0.02,
            trend="up",
            indicators={},
            market_regime="bull",
            liquidity=1000000.0,
            momentum=0.05,
            sentiment=0.7,
            support_levels=[],
            resistance_levels=[],
            market_depth={},
            correlation_matrix={},
            market_impact=0.001,
            volume_profile={},
        )
        market_state_manager.add_state(state)

    # Проверяем, что сохранилось только lookback_period состояний
    assert len(market_state_manager.states) == market_state_manager.lookback_period


def test_market_regime_detection(market_state_manager) -> None:
    """Тест определения режима рынка"""
    # Создаем состояния для разных режимов
    states = [
        # Бычий тренд
        MarketState(
            timestamp=datetime.now(),
            price=100.0 + i,
            volume=1000.0,
            volatility=0.02,
            trend="up",
            indicators={},
            market_regime="unknown",
            liquidity=1000000.0,
            momentum=0.05,
            sentiment=0.7,
            support_levels=[],
            resistance_levels=[],
            market_depth={},
            correlation_matrix={},
            market_impact=0.001,
            volume_profile={},
        )
        for i in range(5)
    ]

    for state in states:
        market_state_manager.add_state(state)

    latest_state = market_state_manager.get_latest_state()
    assert latest_state.market_regime == "bull"


def test_support_resistance_levels(market_state_manager) -> None:
    """Тест расчета уровней поддержки и сопротивления"""
    # Создаем состояния с колебаниями цены
    prices = [100.0, 95.0, 105.0, 90.0, 110.0]
    for price in prices:
        state = MarketState(
            timestamp=datetime.now(),
            price=price,
            volume=1000.0,
            volatility=0.02,
            trend="up",
            indicators={},
            market_regime="unknown",
            liquidity=1000000.0,
            momentum=0.05,
            sentiment=0.7,
            support_levels=[],
            resistance_levels=[],
            market_depth={},
            correlation_matrix={},
            market_impact=0.001,
            volume_profile={},
        )
        market_state_manager.add_state(state)

    latest_state = market_state_manager.get_latest_state()
    assert len(latest_state.support_levels) > 0
    assert len(latest_state.resistance_levels) > 0
    assert all(level <= latest_state.price for level in latest_state.support_levels)
    assert all(level >= latest_state.price for level in latest_state.resistance_levels)


def test_correlation_matrix(market_state_manager) -> None:
    """Тест расчета матрицы корреляций"""
    # Создаем состояния с коррелированными ценами
    for i in range(5):
        state = MarketState(
            timestamp=datetime.now(),
            price=100.0 + i,
            volume=1000.0,
            volatility=0.02,
            trend="up",
            indicators={},
            market_regime="unknown",
            liquidity=1000000.0,
            momentum=0.05,
            sentiment=0.7,
            support_levels=[],
            resistance_levels=[],
            market_depth={},
            correlation_matrix={
                "BTC/USD": {"ETH/USD": 100.0 + i * 0.8},
                "ETH/USD": {"BTC/USD": 100.0 + i},
            },
            market_impact=0.001,
            volume_profile={},
        )
        market_state_manager.add_state(state)

    latest_state = market_state_manager.get_latest_state()
    assert "BTC/USD" in latest_state.correlation_matrix
    assert "ETH/USD" in latest_state.correlation_matrix["BTC/USD"]


def test_market_metrics(market_state_manager) -> None:
    """Тест расчета рыночных метрик"""
    # Создаем состояния с разными метриками
    for i in range(5):
        state = MarketState(
            timestamp=datetime.now(),
            price=100.0 + i,
            volume=1000.0 + i * 100,
            volatility=0.02 + i * 0.001,
            trend="up",
            indicators={},
            market_regime="unknown",
            liquidity=1000000.0,
            momentum=0.05 + i * 0.01,
            sentiment=0.7,
            support_levels=[],
            resistance_levels=[],
            market_depth={},
            correlation_matrix={},
            market_impact=0.001,
            volume_profile={},
        )
        market_state_manager.add_state(state)

    metrics = market_state_manager.get_market_metrics()
    assert "volatility" in metrics
    assert "momentum" in metrics
    assert "trend_strength" in metrics
    assert "volume_trend" in metrics
    assert "market_regime_score" in metrics
    assert "liquidity_score" in metrics
    assert "sentiment_score" in metrics
