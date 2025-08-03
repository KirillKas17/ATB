"""
Unit тесты для liquidity_gravity.

Покрывает:
- Протоколы OrderBookProtocol и LiquidityGravityProtocol
- Конфигурацию LiquidityGravityConfig
- Структуры данных LiquidityGravityResult и OrderBookSnapshot
- Основную модель LiquidityGravityModel
- Все методы вычисления гравитации ликвидности
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch

from domain.market.liquidity_gravity import (
    OrderBookProtocol,
    LiquidityGravityProtocol,
    LiquidityGravityConfig,
    LiquidityGravityResult,
    OrderBookSnapshot,
    LiquidityGravityModel,
    compute_liquidity_gravity
)
from domain.market.market_types import MarketMetadataDict


class TestOrderBookProtocol:
    """Тесты для OrderBookProtocol."""

    @pytest.fixture
    def mock_orderbook(self) -> OrderBookProtocol:
        """Мок реализации OrderBookProtocol."""
        orderbook = Mock(spec=OrderBookProtocol)
        orderbook.bids = [(50000.0, 1.0), (49999.0, 2.0)]
        orderbook.asks = [(50001.0, 1.5), (50002.0, 2.5)]
        orderbook.timestamp = datetime.now()
        orderbook.symbol = "BTC/USD"
        orderbook.get_bid_volume = Mock(return_value=3.0)
        orderbook.get_ask_volume = Mock(return_value=4.0)
        orderbook.get_mid_price = Mock(return_value=50000.5)
        orderbook.get_spread = Mock(return_value=1.0)
        orderbook.get_spread_percentage = Mock(return_value=0.002)
        return orderbook

    def test_protocol_definition(self):
        """Тест определения протокола."""
        assert hasattr(OrderBookProtocol, 'bids')
        assert hasattr(OrderBookProtocol, 'asks')
        assert hasattr(OrderBookProtocol, 'timestamp')
        assert hasattr(OrderBookProtocol, 'symbol')
        assert hasattr(OrderBookProtocol, 'get_bid_volume')
        assert hasattr(OrderBookProtocol, 'get_ask_volume')
        assert hasattr(OrderBookProtocol, 'get_mid_price')
        assert hasattr(OrderBookProtocol, 'get_spread')
        assert hasattr(OrderBookProtocol, 'get_spread_percentage')

    def test_runtime_checkable(self):
        """Тест что протокол является runtime_checkable."""
        assert hasattr(OrderBookProtocol, '__runtime_checkable__')
        assert OrderBookProtocol.__runtime_checkable__ is True


class TestLiquidityGravityProtocol:
    """Тесты для LiquidityGravityProtocol."""

    @pytest.fixture
    def mock_liquidity_gravity(self) -> LiquidityGravityProtocol:
        """Мок реализации LiquidityGravityProtocol."""
        gravity = Mock(spec=LiquidityGravityProtocol)
        gravity.compute_liquidity_gravity = Mock(return_value=0.75)
        gravity.analyze_liquidity_gravity = Mock()
        gravity.compute_gravity_gradient = Mock(return_value={"x": 0.1, "y": 0.2})
        gravity.get_model_statistics = Mock(return_value={"total_calculations": 100})
        gravity.update_config = Mock()
        return gravity

    def test_protocol_definition(self):
        """Тест определения протокола."""
        assert hasattr(LiquidityGravityProtocol, 'compute_liquidity_gravity')
        assert hasattr(LiquidityGravityProtocol, 'analyze_liquidity_gravity')
        assert hasattr(LiquidityGravityProtocol, 'compute_gravity_gradient')
        assert hasattr(LiquidityGravityProtocol, 'get_model_statistics')
        assert hasattr(LiquidityGravityProtocol, 'update_config')

    def test_runtime_checkable(self):
        """Тест что протокол является runtime_checkable."""
        assert hasattr(LiquidityGravityProtocol, '__runtime_checkable__')
        assert LiquidityGravityProtocol.__runtime_checkable__ is True


class TestLiquidityGravityConfig:
    """Тесты для LiquidityGravityConfig."""

    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = LiquidityGravityConfig()
        
        assert config.gravitational_constant == 1e-6
        assert config.min_volume_threshold == 0.001
        assert config.max_price_distance == 0.1
        assert config.volume_weight == 1.0
        assert config.price_weight == 1.0
        assert config.decay_factor == 0.95
        assert config.normalization_factor == 1e6
        assert config.volume_imbalance_threshold == 0.5
        assert config.spread_threshold == 1.0
        assert config.momentum_weight == 0.3
        assert config.volatility_weight == 0.2

    def test_custom_config(self):
        """Тест пользовательской конфигурации."""
        config = LiquidityGravityConfig(
            gravitational_constant=2e-6,
            min_volume_threshold=0.002,
            max_price_distance=0.2,
            volume_weight=1.5,
            price_weight=0.8,
            decay_factor=0.9,
            normalization_factor=2e6,
            risk_thresholds={"low": 0.2, "medium": 0.6, "high": 1.2, "extreme": 2.5},
            volume_imbalance_threshold=0.6,
            spread_threshold=1.5,
            momentum_weight=0.4,
            volatility_weight=0.3
        )
        
        assert config.gravitational_constant == 2e-6
        assert config.min_volume_threshold == 0.002
        assert config.max_price_distance == 0.2
        assert config.volume_weight == 1.5
        assert config.price_weight == 0.8
        assert config.decay_factor == 0.9
        assert config.normalization_factor == 2e6
        assert config.risk_thresholds["low"] == 0.2
        assert config.volume_imbalance_threshold == 0.6
        assert config.spread_threshold == 1.5
        assert config.momentum_weight == 0.4
        assert config.volatility_weight == 0.3

    def test_risk_thresholds_default(self):
        """Тест рисковых порогов по умолчанию."""
        config = LiquidityGravityConfig()
        
        expected_thresholds = {"low": 0.1, "medium": 0.5, "high": 1.0, "extreme": 2.0}
        assert config.risk_thresholds == expected_thresholds


class TestLiquidityGravityResult:
    """Тесты для LiquidityGravityResult."""

    @pytest.fixture
    def sample_gravity_result(self) -> LiquidityGravityResult:
        """Тестовый результат гравитации ликвидности."""
        return LiquidityGravityResult(
            total_gravity=0.75,
            bid_ask_forces=[(1.0, 1.5, 0.3), (2.0, 2.5, 0.4)],
            gravity_distribution={"bid": 0.4, "ask": 0.35},
            risk_level="medium",
            timestamp=datetime.now(),
            metadata={"volatility": 0.025, "volume": 1000000.0},
            volume_imbalance=0.2,
            price_momentum=0.1,
            volatility_score=0.3,
            liquidity_score=0.8,
            market_efficiency=0.7
        )

    def test_gravity_result_creation(self, sample_gravity_result):
        """Тест создания результата гравитации."""
        assert sample_gravity_result.total_gravity == 0.75
        assert len(sample_gravity_result.bid_ask_forces) == 2
        assert sample_gravity_result.gravity_distribution["bid"] == 0.4
        assert sample_gravity_result.risk_level == "medium"
        assert sample_gravity_result.volume_imbalance == 0.2
        assert sample_gravity_result.price_momentum == 0.1
        assert sample_gravity_result.volatility_score == 0.3
        assert sample_gravity_result.liquidity_score == 0.8
        assert sample_gravity_result.market_efficiency == 0.7

    def test_to_dict_conversion(self, sample_gravity_result):
        """Тест преобразования в словарь."""
        result_dict = sample_gravity_result.to_dict()
        
        assert result_dict["total_gravity"] == 0.75
        assert result_dict["bid_ask_forces"] == [(1.0, 1.5, 0.3), (2.0, 2.5, 0.4)]
        assert result_dict["gravity_distribution"]["bid"] == 0.4
        assert result_dict["risk_level"] == "medium"
        assert "timestamp" in result_dict
        assert result_dict["metadata"]["volatility"] == 0.025
        assert result_dict["volume_imbalance"] == 0.2
        assert result_dict["price_momentum"] == 0.1
        assert result_dict["volatility_score"] == 0.3
        assert result_dict["liquidity_score"] == 0.8
        assert result_dict["market_efficiency"] == 0.7


class TestOrderBookSnapshot:
    """Тесты для OrderBookSnapshot."""

    @pytest.fixture
    def sample_orderbook(self) -> OrderBookSnapshot:
        """Тестовый снимок ордербука."""
        return OrderBookSnapshot(
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.5), (50002.0, 2.5)],
            timestamp=datetime.now(),
            symbol="BTC/USD"
        )

    def test_orderbook_creation(self, sample_orderbook):
        """Тест создания снимка ордербука."""
        assert len(sample_orderbook.bids) == 2
        assert len(sample_orderbook.asks) == 2
        assert sample_orderbook.symbol == "BTC/USD"
        assert sample_orderbook.bids[0] == (50000.0, 1.0)
        assert sample_orderbook.asks[0] == (50001.0, 1.5)

    def test_get_bid_volume(self, sample_orderbook):
        """Тест получения объема бидов."""
        bid_volume = sample_orderbook.get_bid_volume()
        assert bid_volume == 3.0  # 1.0 + 2.0

    def test_get_ask_volume(self, sample_orderbook):
        """Тест получения объема асков."""
        ask_volume = sample_orderbook.get_ask_volume()
        assert ask_volume == 4.0  # 1.5 + 2.5

    def test_get_mid_price(self, sample_orderbook):
        """Тест получения средней цены."""
        mid_price = sample_orderbook.get_mid_price()
        expected_mid = (50000.0 + 50001.0) / 2
        assert mid_price == expected_mid

    def test_get_mid_price_empty_orderbook(self):
        """Тест получения средней цены для пустого ордербука."""
        empty_orderbook = OrderBookSnapshot()
        mid_price = empty_orderbook.get_mid_price()
        assert mid_price == 0.0

    def test_get_spread(self, sample_orderbook):
        """Тест получения спреда."""
        spread = sample_orderbook.get_spread()
        expected_spread = 50001.0 - 50000.0
        assert spread == expected_spread

    def test_get_spread_empty_orderbook(self):
        """Тест получения спреда для пустого ордербука."""
        empty_orderbook = OrderBookSnapshot()
        spread = empty_orderbook.get_spread()
        assert spread == 0.0

    def test_get_spread_percentage(self, sample_orderbook):
        """Тест получения спреда в процентах."""
        spread_percentage = sample_orderbook.get_spread_percentage()
        expected_spread = 50001.0 - 50000.0
        expected_mid = (50000.0 + 50001.0) / 2
        expected_percentage = (expected_spread / expected_mid) * 100
        assert abs(spread_percentage - expected_percentage) < 1e-6

    def test_get_spread_percentage_zero_mid_price(self):
        """Тест получения спреда в процентах при нулевой средней цене."""
        orderbook = OrderBookSnapshot(
            bids=[(0.0, 1.0)],
            asks=[(0.0, 1.0)]
        )
        spread_percentage = orderbook.get_spread_percentage()
        assert spread_percentage == 0.0

    def test_empty_orderbook_defaults(self):
        """Тест значений по умолчанию для пустого ордербука."""
        empty_orderbook = OrderBookSnapshot()
        
        assert empty_orderbook.bids == []
        assert empty_orderbook.asks == []
        assert empty_orderbook.symbol == ""
        assert isinstance(empty_orderbook.timestamp, datetime)


class TestLiquidityGravityModel:
    """Тесты для LiquidityGravityModel."""

    @pytest.fixture
    def gravity_model(self) -> LiquidityGravityModel:
        """Экземпляр модели гравитации ликвидности."""
        return LiquidityGravityModel()

    @pytest.fixture
    def custom_gravity_model(self) -> LiquidityGravityModel:
        """Экземпляр модели с пользовательской конфигурацией."""
        config = LiquidityGravityConfig(
            gravitational_constant=2e-6,
            min_volume_threshold=0.002,
            max_price_distance=0.2
        )
        return LiquidityGravityModel(config)

    @pytest.fixture
    def sample_orderbook(self) -> OrderBookSnapshot:
        """Тестовый ордербук."""
        return OrderBookSnapshot(
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.5), (50002.0, 2.5)],
            timestamp=datetime.now(),
            symbol="BTC/USD"
        )

    def test_model_initialization(self):
        """Тест инициализации модели."""
        model = LiquidityGravityModel()
        
        assert model.config is not None
        assert model.config.gravitational_constant == 1e-6
        assert model._total_calculations == 0
        assert model._total_gravity == 0.0
        assert model._min_gravity == float('inf')
        assert model._max_gravity == float('-inf')

    def test_model_initialization_with_config(self):
        """Тест инициализации модели с конфигурацией."""
        config = LiquidityGravityConfig(gravitational_constant=2e-6)
        model = LiquidityGravityModel(config)
        
        assert model.config.gravitational_constant == 2e-6

    def test_compute_liquidity_gravity(self, gravity_model, sample_orderbook):
        """Тест вычисления гравитации ликвидности."""
        gravity = gravity_model.compute_liquidity_gravity(sample_orderbook)
        
        assert isinstance(gravity, float)
        assert gravity >= 0.0
        assert gravity_model._total_calculations == 1

    def test_compute_liquidity_gravity_empty_orderbook(self, gravity_model):
        """Тест вычисления гравитации для пустого ордербука."""
        empty_orderbook = OrderBookSnapshot()
        gravity = gravity_model.compute_liquidity_gravity(empty_orderbook)
        
        assert gravity == 0.0

    def test_compute_liquidity_gravity_custom_config(self, custom_gravity_model, sample_orderbook):
        """Тест вычисления гравитации с пользовательской конфигурацией."""
        gravity = custom_gravity_model.compute_liquidity_gravity(sample_orderbook)
        
        assert isinstance(gravity, float)
        assert gravity >= 0.0

    def test_compute_gravitational_force(self, gravity_model):
        """Тест вычисления гравитационной силы."""
        force = gravity_model._compute_gravitational_force(1.0, 2.0, 0.01)
        
        assert isinstance(force, float)
        assert force >= 0.0

    def test_compute_gravitational_force_zero_distance(self, gravity_model):
        """Тест вычисления гравитационной силы при нулевом расстоянии."""
        force = gravity_model._compute_gravitational_force(1.0, 2.0, 0.0)
        
        # При нулевом расстоянии сила должна быть максимальной
        assert force > 0.0

    def test_analyze_liquidity_gravity(self, gravity_model, sample_orderbook):
        """Тест анализа гравитации ликвидности."""
        result = gravity_model.analyze_liquidity_gravity(sample_orderbook)
        
        assert isinstance(result, LiquidityGravityResult)
        assert result.total_gravity >= 0.0
        assert isinstance(result.bid_ask_forces, list)
        assert isinstance(result.gravity_distribution, dict)
        assert result.risk_level in ["low", "medium", "high", "extreme"]
        assert result.timestamp == sample_orderbook.timestamp
        assert isinstance(result.metadata, dict)
        assert isinstance(result.volume_imbalance, float)
        assert isinstance(result.price_momentum, float)
        assert isinstance(result.volatility_score, float)
        assert isinstance(result.liquidity_score, float)
        assert isinstance(result.market_efficiency, float)

    def test_determine_risk_level(self, gravity_model, sample_orderbook):
        """Тест определения уровня риска."""
        # Тест для низкого риска
        low_risk = gravity_model._determine_risk_level(0.05, sample_orderbook)
        assert low_risk == "low"
        
        # Тест для среднего риска
        medium_risk = gravity_model._determine_risk_level(0.3, sample_orderbook)
        assert medium_risk == "medium"
        
        # Тест для высокого риска
        high_risk = gravity_model._determine_risk_level(0.8, sample_orderbook)
        assert high_risk == "high"
        
        # Тест для экстремального риска
        extreme_risk = gravity_model._determine_risk_level(1.5, sample_orderbook)
        assert extreme_risk == "extreme"

    def test_compute_gravity_gradient(self, gravity_model, sample_orderbook):
        """Тест вычисления градиента гравитации."""
        gradient = gravity_model.compute_gravity_gradient(sample_orderbook)
        
        assert isinstance(gradient, dict)
        assert "bid_gradient" in gradient
        assert "ask_gradient" in gradient
        assert "total_gradient" in gradient
        assert all(isinstance(v, float) for v in gradient.values())

    def test_calculate_volume_imbalance(self, gravity_model, sample_orderbook):
        """Тест расчета дисбаланса объема."""
        imbalance = gravity_model._calculate_volume_imbalance(sample_orderbook)
        
        assert isinstance(imbalance, float)
        assert -1.0 <= imbalance <= 1.0

    def test_calculate_price_momentum(self, gravity_model, sample_orderbook):
        """Тест расчета ценового импульса."""
        momentum = gravity_model._calculate_price_momentum(sample_orderbook)
        
        assert isinstance(momentum, float)

    def test_calculate_volatility_score(self, gravity_model, sample_orderbook):
        """Тест расчета показателя волатильности."""
        volatility = gravity_model._calculate_volatility_score(sample_orderbook)
        
        assert isinstance(volatility, float)
        assert 0.0 <= volatility <= 1.0

    def test_calculate_liquidity_score(self, gravity_model, sample_orderbook):
        """Тест расчета показателя ликвидности."""
        liquidity = gravity_model._calculate_liquidity_score(sample_orderbook)
        
        assert isinstance(liquidity, float)
        assert 0.0 <= liquidity <= 1.0

    def test_calculate_market_efficiency(self, gravity_model, sample_orderbook):
        """Тест расчета эффективности рынка."""
        efficiency = gravity_model._calculate_market_efficiency(sample_orderbook)
        
        assert isinstance(efficiency, float)
        assert 0.0 <= efficiency <= 1.0

    def test_update_statistics(self, gravity_model):
        """Тест обновления статистики."""
        initial_calculations = gravity_model._total_calculations
        initial_gravity = gravity_model._total_gravity
        
        gravity_model._update_statistics(0.5)
        
        assert gravity_model._total_calculations == initial_calculations + 1
        assert gravity_model._total_gravity == initial_gravity + 0.5
        assert gravity_model._min_gravity <= 0.5
        assert gravity_model._max_gravity >= 0.5

    def test_get_model_statistics(self, gravity_model, sample_orderbook):
        """Тест получения статистики модели."""
        # Выполняем несколько вычислений
        gravity_model.compute_liquidity_gravity(sample_orderbook)
        gravity_model.compute_liquidity_gravity(sample_orderbook)
        
        stats = gravity_model.get_model_statistics()
        
        assert isinstance(stats, dict)
        assert "total_calculations" in stats
        assert "total_gravity" in stats
        assert "average_gravity" in stats
        assert "min_gravity" in stats
        assert "max_gravity" in stats
        assert stats["total_calculations"] == 2

    def test_update_config(self, gravity_model):
        """Тест обновления конфигурации."""
        old_constant = gravity_model.config.gravitational_constant
        new_config = LiquidityGravityConfig(gravitational_constant=2e-6)
        
        gravity_model.update_config(new_config)
        
        assert gravity_model.config.gravitational_constant == 2e-6
        assert gravity_model.config.gravitational_constant != old_constant

    def test_protocol_compliance(self, gravity_model):
        """Тест соответствия протоколу."""
        assert isinstance(gravity_model, LiquidityGravityProtocol)


class TestComputeLiquidityGravityFunction:
    """Тесты для функции compute_liquidity_gravity."""

    @pytest.fixture
    def sample_orderbook(self) -> OrderBookSnapshot:
        """Тестовый ордербук."""
        return OrderBookSnapshot(
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.5), (50002.0, 2.5)],
            timestamp=datetime.now(),
            symbol="BTC/USD"
        )

    def test_compute_liquidity_gravity_function(self, sample_orderbook):
        """Тест функции compute_liquidity_gravity."""
        gravity = compute_liquidity_gravity(sample_orderbook)
        
        assert isinstance(gravity, float)
        assert gravity >= 0.0

    def test_compute_liquidity_gravity_empty_orderbook(self):
        """Тест функции для пустого ордербука."""
        empty_orderbook = OrderBookSnapshot()
        gravity = compute_liquidity_gravity(empty_orderbook)
        
        assert gravity == 0.0

    def test_compute_liquidity_gravity_with_mock_orderbook(self):
        """Тест функции с мок ордербуком."""
        mock_orderbook = Mock(spec=OrderBookProtocol)
        mock_orderbook.bids = [(50000.0, 1.0)]
        mock_orderbook.asks = [(50001.0, 1.5)]
        mock_orderbook.get_bid_volume = Mock(return_value=1.0)
        mock_orderbook.get_ask_volume = Mock(return_value=1.5)
        mock_orderbook.get_mid_price = Mock(return_value=50000.5)
        mock_orderbook.get_spread = Mock(return_value=1.0)
        mock_orderbook.get_spread_percentage = Mock(return_value=0.002)
        
        gravity = compute_liquidity_gravity(mock_orderbook)
        
        assert isinstance(gravity, float)
        assert gravity >= 0.0


class TestLiquidityGravityIntegration:
    """Интеграционные тесты для гравитации ликвидности."""

    @pytest.fixture
    def gravity_model(self) -> LiquidityGravityModel:
        """Экземпляр модели гравитации ликвидности."""
        return LiquidityGravityModel()

    def test_full_analysis_workflow(self, gravity_model):
        """Тест полного рабочего процесса анализа."""
        # Создаем тестовый ордербук
        orderbook = OrderBookSnapshot(
            bids=[(50000.0, 1.0), (49999.0, 2.0), (49998.0, 3.0)],
            asks=[(50001.0, 1.5), (50002.0, 2.5), (50003.0, 3.5)],
            timestamp=datetime.now(),
            symbol="BTC/USD"
        )
        
        # Выполняем полный анализ
        result = gravity_model.analyze_liquidity_gravity(orderbook)
        
        # Проверяем результат
        assert isinstance(result, LiquidityGravityResult)
        assert result.total_gravity > 0.0
        assert len(result.bid_ask_forces) > 0
        assert "bid" in result.gravity_distribution
        assert "ask" in result.gravity_distribution
        assert result.risk_level in ["low", "medium", "high", "extreme"]
        
        # Проверяем статистику модели
        stats = gravity_model.get_model_statistics()
        assert stats["total_calculations"] > 0
        assert stats["total_gravity"] > 0.0

    def test_multiple_calculations_consistency(self, gravity_model):
        """Тест консистентности множественных вычислений."""
        orderbook = OrderBookSnapshot(
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 1.5)],
            timestamp=datetime.now(),
            symbol="BTC/USD"
        )
        
        # Выполняем несколько вычислений
        gravity1 = gravity_model.compute_liquidity_gravity(orderbook)
        gravity2 = gravity_model.compute_liquidity_gravity(orderbook)
        
        # Результаты должны быть одинаковыми для одинаковых данных
        assert abs(gravity1 - gravity2) < 1e-6

    def test_configuration_impact(self):
        """Тест влияния конфигурации на результаты."""
        # Базовая конфигурация
        base_config = LiquidityGravityConfig()
        base_model = LiquidityGravityModel(base_config)
        
        # Конфигурация с увеличенной гравитационной постоянной
        high_gravity_config = LiquidityGravityConfig(gravitational_constant=2e-6)
        high_gravity_model = LiquidityGravityModel(high_gravity_config)
        
        orderbook = OrderBookSnapshot(
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 1.5)],
            timestamp=datetime.now(),
            symbol="BTC/USD"
        )
        
        base_gravity = base_model.compute_liquidity_gravity(orderbook)
        high_gravity = high_gravity_model.compute_liquidity_gravity(orderbook)
        
        # Результаты должны отличаться
        assert base_gravity != high_gravity 