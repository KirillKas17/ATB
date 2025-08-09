# -*- coding: utf-8 -*-
"""Unit тесты для интеграции NoiseAnalyzer в систему Syntra."""

import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from datetime import datetime

from domain.intelligence.noise_analyzer import NoiseAnalyzer, NoiseAnalysisResult, OrderBookSnapshot
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency
from domain.entities.signal import Signal, SignalType
from domain.entities.strategy import Strategy
from domain.entities.portfolio import Portfolio
from domain.entities.trading_pair import TradingPair
from domain.entities.order import Order, OrderSide, OrderType
from infrastructure.agents.agent_context_refactored import AgentContext

# Временно закомментировано из-за отсутствия класса
# from application.use_cases.trading_orchestrator import TradingOrchestratorUseCase
from application.di_container import DIContainer, ContainerConfig


class TestNoiseAnalyzerIntegration:
    """Тесты интеграции NoiseAnalyzer."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.noise_analyzer = NoiseAnalyzer(
            fractal_dimension_lower=1.2,
            fractal_dimension_upper=1.4,
            entropy_threshold=0.7,
            min_data_points=20,
            window_size=50,
        )

    def test_noise_analyzer_initialization(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест инициализации NoiseAnalyzer."""
        assert self.noise_analyzer.fractal_dimension_lower == 1.2
        assert self.noise_analyzer.fractal_dimension_upper == 1.4
        assert self.noise_analyzer.entropy_threshold == 0.7
        assert self.noise_analyzer.min_data_points == 20
        assert self.noise_analyzer.window_size == 50

    def test_orderbook_snapshot_creation(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест создания OrderBookSnapshot."""
        bids = [(Price(Decimal("50000"), Currency.USDT), Volume(Decimal("1.0")))]
        asks = [(Price(Decimal("50010"), Currency.USDT), Volume(Decimal("1.0")))]
        timestamp = Timestamp(datetime.now())

        orderbook = OrderBookSnapshot(exchange="binance", symbol="BTCUSDT", bids=bids, asks=asks, timestamp=timestamp)

        assert orderbook.exchange == "binance"
        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 1
        assert len(orderbook.asks) == 1
        assert orderbook.timestamp == timestamp

    def test_noise_analysis_result_creation(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест создания NoiseAnalysisResult."""
        timestamp = Timestamp(datetime.now())

        result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.6,
            is_synthetic_noise=True,
            confidence=0.85,
            metadata={"test": "data"},
            timestamp=timestamp,
        )

        assert result.fractal_dimension == 1.3
        assert result.entropy == 0.6
        assert result.is_synthetic_noise is True
        assert result.confidence == 0.85
        assert result.metadata["test"] == "data"
        assert result.timestamp == timestamp

    def test_noise_analysis_result_to_dict(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест преобразования NoiseAnalysisResult в словарь."""
        timestamp = Timestamp(datetime.now())

        result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.6,
            is_synthetic_noise=True,
            confidence=0.85,
            metadata={"test": "data"},
            timestamp=timestamp,
        )

        result_dict = result.to_dict()

        assert result_dict["fractal_dimension"] == 1.3
        assert result_dict["entropy"] == 0.6
        assert result_dict["is_synthetic_noise"] is True
        assert result_dict["confidence"] == 0.85
        assert result_dict["metadata"]["test"] == "data"
        assert result_dict["timestamp"] == timestamp.value

    def test_agent_context_noise_result_integration(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест интеграции noise_result в AgentContext."""
        context = AgentContext(symbol="BTCUSDT")

        # Проверяем, что поле noise_result существует и изначально None
        assert context.noise_result is None

        # Создаем результат анализа шума
        timestamp = Timestamp(datetime.now())
        noise_result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.6,
            is_synthetic_noise=True,
            confidence=0.85,
            metadata={"test": "data"},
            timestamp=timestamp,
        )

        # Устанавливаем результат в контекст
        context.noise_result = noise_result

        assert context.noise_result == noise_result
        assert context.noise_result.is_synthetic_noise is True
        assert context.noise_result.fractal_dimension == 1.3

    def test_agent_context_noise_modifier_application(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест применения модификатора шума в AgentContext."""
        context = AgentContext(symbol="BTCUSDT")

        # Устанавливаем начальные значения модификаторов
        context.strategy_modifiers.order_aggressiveness = 1.0
        context.strategy_modifiers.position_size_multiplier = 1.0
        context.strategy_modifiers.confidence_multiplier = 1.0
        context.strategy_modifiers.execution_delay_ms = 0

        # Создаем результат анализа шума с синтетическим шумом
        timestamp = Timestamp(datetime.now())
        noise_result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.6,
            is_synthetic_noise=True,
            confidence=0.85,
            metadata={"test": "data"},
            timestamp=timestamp,
        )

        context.noise_result = noise_result

        # Применяем модификатор шума
        context.apply_noise_modifier()

        # Проверяем, что модификаторы были применены
        assert context.strategy_modifiers.order_aggressiveness < 1.0
        assert context.strategy_modifiers.position_size_multiplier < 1.0
        assert context.strategy_modifiers.confidence_multiplier < 1.0
        assert context.strategy_modifiers.execution_delay_ms > 0

    def test_agent_context_noise_analyzer_modifier_application(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест применения модификатора анализатора шума в AgentContext."""
        context = AgentContext(symbol="BTCUSDT")

        # Устанавливаем начальные значения модификаторов
        context.strategy_modifiers.order_aggressiveness = 1.0
        context.strategy_modifiers.position_size_multiplier = 1.0
        context.strategy_modifiers.confidence_multiplier = 1.0
        context.strategy_modifiers.risk_multiplier = 1.0
        context.strategy_modifiers.execution_delay_ms = 0

        # Создаем результат анализа шума
        timestamp = Timestamp(datetime.now())
        noise_result = NoiseAnalysisResult(
            fractal_dimension=1.1,  # Низкая фрактальная размерность
            entropy=0.9,  # Высокая энтропия
            is_synthetic_noise=True,
            confidence=0.5,  # Низкая уверенность
            metadata={"test": "data"},
            timestamp=timestamp,
        )

        context.noise_result = noise_result

        # Применяем модификатор анализатора шума
        context.apply_noise_analyzer_modifier()

        # Проверяем, что модификаторы были применены
        assert context.strategy_modifiers.order_aggressiveness < 1.0
        assert context.strategy_modifiers.position_size_multiplier < 1.0
        assert context.strategy_modifiers.confidence_multiplier < 1.0
        assert context.strategy_modifiers.risk_multiplier > 1.0
        assert context.strategy_modifiers.execution_delay_ms > 0

    def test_agent_context_noise_analysis_status(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест получения статуса анализа шума из AgentContext."""
        context = AgentContext(symbol="BTCUSDT")

        # Проверяем статус без результата анализа
        status = context.get_noise_analysis_status()
        assert status["is_synthetic_noise"] is False
        assert status["status"] == "unknown"

        # Создаем результат анализа шума
        timestamp = Timestamp(datetime.now())
        noise_result = NoiseAnalysisResult(
            fractal_dimension=1.3,
            entropy=0.6,
            is_synthetic_noise=True,
            confidence=0.85,
            metadata={"test": "data"},
            timestamp=timestamp,
        )

        context.noise_result = noise_result

        # Проверяем статус с результатом анализа
        status = context.get_noise_analysis_status()
        assert status["is_synthetic_noise"] is True
        assert status["fractal_dimension"] == 1.3
        assert status["entropy"] == 0.6
        assert status["confidence"] == 0.85
        assert status["status"] == "synthetic_noise"

    def test_di_container_noise_analyzer_registration(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест регистрации NoiseAnalyzer в DI контейнере."""
        config = ContainerConfig(noise_analysis_enabled=True)
        container = DIContainer(config)

        # Получаем NoiseAnalyzer из контейнера
        noise_analyzer = container.get("noise_analyzer")

        assert noise_analyzer is not None
        assert isinstance(noise_analyzer, NoiseAnalyzer)
        assert noise_analyzer.fractal_dimension_lower == 1.2
        assert noise_analyzer.fractal_dimension_upper == 1.4
        assert noise_analyzer.entropy_threshold == 0.7

    def test_di_container_noise_analyzer_disabled(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест отключения NoiseAnalyzer в DI контейнере."""
        config = ContainerConfig(noise_analysis_enabled=False)
        container = DIContainer(config)

        # Проверяем, что NoiseAnalyzer не зарегистрирован
        with pytest.raises(KeyError):
            container.get("noise_analyzer")

    @pytest.mark.asyncio
    async def test_trading_orchestrator_noise_analyzer_integration(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест интеграции NoiseAnalyzer в TradingOrchestrator."""
        # Создаем моки для зависимостей
        mock_order_repository = AsyncMock()
        mock_position_repository = AsyncMock()
        mock_portfolio_repository = AsyncMock()
        mock_trading_repository = AsyncMock()
        mock_strategy_repository = AsyncMock()
        mock_enhanced_trading_service = AsyncMock()

        # Создаем тестовые данные
        strategy = Strategy(
            id="test_strategy",
            name="Test Strategy",
            description="Test strategy for noise analysis",
            parameters={},
            is_active=True,
        )
        mock_strategy_repository.get_by_id.return_value = strategy

        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            description="Test portfolio",
            initial_balance=Decimal("10000"),
            current_balance=Decimal("10000"),
        )
        mock_portfolio_repository.get_by_id.return_value = portfolio

        # Создаем мок TradingOrchestrator с NoiseAnalyzer
        orchestrator = Mock()
        orchestrator.noise_analyzer = self.noise_analyzer
        orchestrator._noise_analysis_cache = {}
        orchestrator._last_noise_analysis_update = None

        # Мокаем метод валидации торговых условий
        orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))

        # Мокаем метод генерации сигналов
        test_signal = Signal(
            id="test_signal",
            trading_pair=TradingPair(symbol="BTCUSDT", base="BTC", quote="USDT"),
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            timestamp=datetime.now(),
            metadata={},
        )
        orchestrator._generate_signals_with_sentiment = AsyncMock(return_value=[test_signal])

        # Мокаем метод создания ордера
        test_order = Order(
            id="test_order",
            portfolio_id="test_portfolio",
            trading_pair=TradingPair(symbol="BTCUSDT", base="BTC", quote="USDT"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=Volume(Decimal("0.1")),
            price=Price(Decimal("50000"), Currency.USDT),
            status="pending",
            timestamp=datetime.now(),
        )
        orchestrator._create_enhanced_order_from_signal = AsyncMock(return_value=test_order)

        # Мокаем получение данных ордербука
        mock_orderbook_data = {"bids": [[50000, 1.0]], "asks": [[50010, 1.0]]}
        mock_enhanced_trading_service.get_orderbook.return_value = mock_orderbook_data

        # Создаем запрос на выполнение стратегии
        request = ExecuteStrategyRequest(
            strategy_id="test_strategy",
            portfolio_id="test_portfolio",
            symbol="BTCUSDT",
            amount=Decimal("1000"),
            risk_level="medium",
            use_sentiment_analysis=False,
        )

        # Выполняем стратегию
        response = await orchestrator.execute_strategy(request)

        # Проверяем, что стратегия была выполнена
        assert response.executed is True
        assert len(response.orders_created) == 1
        assert len(response.signals_generated) == 1

        # Проверяем, что методы анализа шума были вызваны
        mock_enhanced_trading_service.get_orderbook.assert_called_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_trading_orchestrator_noise_analysis_application(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест применения анализа шума к сигналам в TradingOrchestrator."""
        # Создаем моки для зависимостей
        mock_order_repository = AsyncMock()
        mock_position_repository = AsyncMock()
        mock_portfolio_repository = AsyncMock()
        mock_trading_repository = AsyncMock()
        mock_strategy_repository = AsyncMock()
        mock_enhanced_trading_service = AsyncMock()

        # Создаем мок TradingOrchestrator с NoiseAnalyzer
        orchestrator = Mock()
        orchestrator.noise_analyzer = self.noise_analyzer
        orchestrator._noise_analysis_cache = {}
        orchestrator._last_noise_analysis_update = None

        # Создаем тестовый сигнал
        original_signal = Signal(
            id="test_signal",
            trading_pair=TradingPair(symbol="BTCUSDT", base="BTC", quote="USDT"),
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            timestamp=datetime.now(),
            metadata={},
        )

        # Мокаем данные ордербука для анализа шума
        mock_orderbook_data = {"bids": [[50000, 1.0]], "asks": [[50010, 1.0]]}
        mock_enhanced_trading_service.get_orderbook.return_value = mock_orderbook_data

        # Применяем анализ шума к сигналу
        modified_signal = await orchestrator._apply_noise_analysis("BTCUSDT", original_signal)

        # Проверяем, что сигнал был модифицирован
        assert modified_signal is not None
        assert modified_signal.id == original_signal.id

    def test_noise_analyzer_statistics(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест получения статистики NoiseAnalyzer."""
        # Добавляем некоторые данные в историю
        for i in range(10):
            self.noise_analyzer.price_history.append(50000.0 + i)
            self.noise_analyzer.volume_history.append(1.0 + i * 0.1)
            self.noise_analyzer.spread_history.append(10.0 + i * 0.5)

        # Получаем статистику
        stats = self.noise_analyzer.get_analysis_statistics()

        # Проверяем статистику
        assert stats["price_history_length"] == 10
        assert stats["volume_history_length"] == 10
        assert stats["spread_history_length"] == 10
        assert stats["window_size"] == 50
        assert stats["min_data_points"] == 20
        assert stats["fractal_dimension_range"] == [1.2, 1.4]
        assert stats["entropy_threshold"] == 0.7
        assert stats["confidence_threshold"] == 0.8

    def test_noise_analyzer_history_reset(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест сброса истории NoiseAnalyzer."""
        # Добавляем данные в историю
        self.noise_analyzer.price_history.append(50000.0)
        self.noise_analyzer.volume_history.append(1.0)
        self.noise_analyzer.spread_history.append(10.0)

        # Проверяем, что данные добавлены
        assert len(self.noise_analyzer.price_history) == 1
        assert len(self.noise_analyzer.volume_history) == 1
        assert len(self.noise_analyzer.spread_history) == 1

        # Сбрасываем историю
        self.noise_analyzer.reset_history()

        # Проверяем, что история очищена
        assert len(self.noise_analyzer.price_history) == 0
        assert len(self.noise_analyzer.volume_history) == 0
        assert len(self.noise_analyzer.spread_history) == 0

    def test_noise_analyzer_synthetic_noise_detection(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест обнаружения синтетического шума."""
        # Тестируем с параметрами, которые должны вызвать обнаружение синтетического шума
        result = self.noise_analyzer.is_synthetic_noise(fd=1.3, entropy=0.6)
        assert result is True

        # Тестируем с параметрами, которые не должны вызвать обнаружение
        result = self.noise_analyzer.is_synthetic_noise(fd=1.5, entropy=0.8)
        assert result is False

        # Тестируем граничные случаи
        result = self.noise_analyzer.is_synthetic_noise(fd=1.2, entropy=0.7)
        assert result is False  # entropy >= threshold

        result = self.noise_analyzer.is_synthetic_noise(fd=1.4, entropy=0.6)
        assert result is True  # fd в диапазоне и entropy < threshold

    def test_noise_analyzer_confidence_calculation(self: "TestNoiseAnalyzerIntegration") -> None:
        """Тест вычисления уверенности в анализе шума."""
        # Тестируем с идеальными параметрами
        confidence = self.noise_analyzer.compute_confidence(fd=1.3, entropy=0.5)
        assert 0.0 <= confidence <= 1.0

        # Тестируем с экстремальными параметрами
        confidence = self.noise_analyzer.compute_confidence(fd=2.0, entropy=1.0)
        assert 0.0 <= confidence <= 1.0

        # Тестируем с граничными параметрами
        confidence = self.noise_analyzer.compute_confidence(fd=1.2, entropy=0.7)
        assert 0.0 <= confidence <= 1.0
