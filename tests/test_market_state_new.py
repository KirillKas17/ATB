from datetime import datetime, timedelta

import pytest

from core.market_state import MarketState, MarketStateManager


@pytest.fixture
def sample_market_state():
    """Фикстура для MarketState"""
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
def market_state_manager():
    """Фикстура для MarketStateManager"""
    config = {
        "lookback_period": 100,
        "regime_threshold": 0.7,
        "volatility_window": 20,
        "momentum_window": 10,
    }
    return MarketStateManager(config)


class TestMarketState:
    """Тесты для MarketState"""

    def test_creation(self, sample_market_state):
        """Тест создания MarketState"""
        assert sample_market_state.price == 100.0
        assert sample_market_state.volume == 1000.0
        assert sample_market_state.market_regime == "bull"
        assert len(sample_market_state.support_levels) == 2
        assert len(sample_market_state.resistance_levels) == 2
        assert "BTC/USD" in sample_market_state.correlation_matrix

    def test_serialization(self, sample_market_state):
        """Тест сериализации/десериализации"""
        # Сериализация
        state_dict = sample_market_state.to_dict()

        # Проверка структуры
        assert "timestamp" in state_dict
        assert "price" in state_dict
        assert "volume" in state_dict
        assert "market_regime" in state_dict
        assert "support_levels" in state_dict
        assert "resistance_levels" in state_dict
        assert "correlation_matrix" in state_dict

        # Десериализация
        new_state = MarketState.from_dict(state_dict)

        # Проверка равенства
        assert new_state.price == sample_market_state.price
        assert new_state.volume == sample_market_state.volume
        assert new_state.market_regime == sample_market_state.market_regime
        assert new_state.support_levels == sample_market_state.support_levels
        assert new_state.resistance_levels == sample_market_state.resistance_levels


class TestMarketStateManager:
    """Тесты для MarketStateManager"""

    def test_initialization(self, market_state_manager):
        """Тест инициализации"""
        assert market_state_manager.lookback_period == 100
        assert market_state_manager.regime_threshold == 0.7
        assert market_state_manager.volatility_window == 20
        assert market_state_manager.momentum_window == 10

    def test_add_state(self, market_state_manager, sample_market_state):
        """Тест добавления состояния"""
        market_state_manager.add_state(sample_market_state)
        latest_state = market_state_manager.get_latest_state()

        assert latest_state is not None
        assert latest_state.price == sample_market_state.price
        assert latest_state.volume == sample_market_state.volume

    def test_lookback(self, market_state_manager):
        """Тест ограничения истории состояний"""
        # Добавляем больше состояний, чем lookback_period
        for i in range(150):
            state = MarketState(
                timestamp=datetime.now() + timedelta(minutes=i),
                price=100.0 + i,
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
            market_state_manager.add_state(state)

        # Проверяем, что сохранилось только lookback_period состояний
        assert len(market_state_manager.states) == market_state_manager.lookback_period

    def test_market_regime_detection(self, market_state_manager):
        """Тест определения рыночного режима"""
        # Создаем состояния для бычьего рынка
        for i in range(20):
            state = MarketState(
                timestamp=datetime.now() + timedelta(minutes=i),
                price=100.0 + i * 0.5,  # Растущие цены
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
            market_state_manager.add_state(state)

        latest_state = market_state_manager.get_latest_state()
        assert latest_state.market_regime == "bull"

    def test_support_resistance_levels(self, market_state_manager):
        """Тест расчета уровней поддержки и сопротивления"""
        # Создаем состояния с колебаниями цены
        prices = [100.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0]
        for i, price in enumerate(prices):
            state = MarketState(
                timestamp=datetime.now() + timedelta(minutes=i),
                price=price,
                volume=1000.0,
                volatility=0.02,
                trend="up",
                indicators={"rsi": 65.0, "macd": 0.5},
                market_regime="bull",
                liquidity=1000000.0,
                momentum=0.05,
                sentiment=0.7,
                support_levels=[],
                resistance_levels=[],
                market_depth={"bid": 500.0, "ask": 500.0},
                correlation_matrix={"BTC/USD": {"ETH/USD": 0.8}},
                market_impact=0.001,
                volume_profile={100.0: 1000.0, 101.0: 800.0},
            )
            market_state_manager.add_state(state)

        latest_state = market_state_manager.get_latest_state()
        assert len(latest_state.support_levels) > 0
        assert len(latest_state.resistance_levels) > 0

    def test_correlation_matrix(self, market_state_manager):
        """Тест расчета корреляционной матрицы"""
        # Создаем состояния для двух коррелированных инструментов
        for i in range(20):
            state = MarketState(
                timestamp=datetime.now() + timedelta(minutes=i),
                price=100.0 + i,
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
                correlation_matrix={
                    "BTC/USD": {"ETH/USD": 0.8 + i * 0.01},
                    "ETH/USD": {"BTC/USD": 0.8 + i * 0.01},
                },
                market_impact=0.001,
                volume_profile={100.0: 1000.0, 101.0: 800.0},
            )
            market_state_manager.add_state(state)

        latest_state = market_state_manager.get_latest_state()
        assert "BTC/USD" in latest_state.correlation_matrix
        assert "ETH/USD" in latest_state.correlation_matrix["BTC/USD"]

    def test_market_metrics(self, market_state_manager):
        """Тест расчета рыночных метрик"""
        # Создаем состояния с разными метриками
        for i in range(20):
            state = MarketState(
                timestamp=datetime.now() + timedelta(minutes=i),
                price=100.0 + i,
                volume=1000.0 + i * 100,
                volatility=0.02 + i * 0.001,
                trend="up",
                indicators={"rsi": 65.0, "macd": 0.5},
                market_regime="bull",
                liquidity=1000000.0 + i * 10000,
                momentum=0.05 + i * 0.001,
                sentiment=0.7 + i * 0.01,
                support_levels=[95.0, 90.0],
                resistance_levels=[105.0, 110.0],
                market_depth={"bid": 500.0, "ask": 500.0},
                correlation_matrix={"BTC/USD": {"ETH/USD": 0.8}},
                market_impact=0.001,
                volume_profile={100.0: 1000.0, 101.0: 800.0},
            )
            market_state_manager.add_state(state)

        latest_state = market_state_manager.get_latest_state()
        assert latest_state.volatility > 0
        assert latest_state.momentum > 0
        assert latest_state.liquidity > 0
        assert latest_state.sentiment > 0
