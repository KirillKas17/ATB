"""
Unit тесты для интеграции MarketPatternRecognizer.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from domain.intelligence.market_pattern_recognizer import MarketPatternRecognizer, PatternDetection, PatternType
from infrastructure.agents.agent_context_refactored import AgentContext
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
from application.di_container import DIContainer


class TestMarketPatternRecognizerIntegration:
    """Тесты интеграции MarketPatternRecognizer."""

    @pytest.fixture
    def market_pattern_recognizer(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра MarketPatternRecognizer."""
        return MarketPatternRecognizer()

    @pytest.fixture
    def agent_context(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание AgentContext."""
        return AgentContext(symbol="BTCUSDT")

    @pytest.fixture
    def mock_repositories(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание моков репозиториев."""
        return {
            "order_repository": Mock(),
            "position_repository": Mock(),
            "portfolio_repository": Mock(),
            "trading_repository": Mock(),
            "strategy_repository": Mock(),
            "enhanced_trading_service": Mock(),
        }

    @pytest.fixture
    def trading_orchestrator(self, mock_repositories, market_pattern_recognizer) -> Any:
        """Создание TradingOrchestrator с MarketPatternRecognizer."""
        return DefaultTradingOrchestratorUseCase(
            **mock_repositories, market_pattern_recognizer=market_pattern_recognizer
        )

    def test_di_container_integration(self: "TestMarketPatternRecognizerIntegration") -> None:
        """Тест интеграции в DI контейнер."""
        config = ContainerConfig(market_pattern_recognition_enabled=True)
        container = DIContainer(config)
        # Проверяем, что MarketPatternRecognizer зарегистрирован
        recognizer = container.get("market_pattern_recognizer")
        assert isinstance(recognizer, MarketPatternRecognizer)

    def test_agent_context_integration(self, agent_context, market_pattern_recognizer) -> None:
        """Тест интеграции в AgentContext."""
        # Создаем тестовый паттерн
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.85,
            strength=0.8,
            direction="up",
            metadata={},
            volume_anomaly=2.5,
            price_impact=0.03,
            order_book_imbalance=0.6,
            spread_widening=0.02,
            depth_absorption=0.7,
        )
        # Устанавливаем паттерн в контекст
        agent_context.market_pattern_result = pattern
        # Применяем модификатор
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что модификаторы применены
        assert agent_context.strategy_modifiers.order_aggressiveness > 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier > 1.0
        assert agent_context.strategy_modifiers.confidence_multiplier > 1.0

    def test_agent_context_whale_absorption_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для поглощения китами."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.9,
            strength=0.9,
            direction="up",
            metadata={},
            volume_anomaly=3.0,
            price_impact=0.05,
            order_book_imbalance=0.8,
            spread_widening=0.03,
            depth_absorption=0.8,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для восходящего поглощения
        assert agent_context.strategy_modifiers.order_aggressiveness > 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier > 1.0

    def test_agent_context_mm_spoofing_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для спуфинга маркет-мейкеров."""
        pattern = PatternDetection(
            pattern_type=PatternType.MM_SPOOFING,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.7,
            direction="down",
            metadata={},
            volume_anomaly=2.0,
            price_impact=0.04,
            order_book_imbalance=0.9,
            spread_widening=0.05,
            depth_absorption=0.6,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для спуфинга (должны снижать агрессивность)
        assert agent_context.strategy_modifiers.order_aggressiveness < 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier < 1.0
        assert agent_context.strategy_modifiers.risk_multiplier > 1.0
        assert agent_context.strategy_modifiers.execution_delay_ms > 0

    def test_agent_context_iceberg_detection_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для обнаружения айсбергов."""
        pattern = PatternDetection(
            pattern_type=PatternType.ICEBERG_DETECTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.7,
            strength=0.6,
            direction="neutral",
            metadata={},
            volume_anomaly=1.8,
            price_impact=0.02,
            order_book_imbalance=0.5,
            spread_widening=0.01,
            depth_absorption=0.4,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для айсбергов
        assert agent_context.strategy_modifiers.order_aggressiveness < 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier < 1.0
        assert agent_context.strategy_modifiers.price_offset_percent > 0.0

    def test_agent_context_liquidity_grab_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для захвата ликвидности."""
        pattern = PatternDetection(
            pattern_type=PatternType.LIQUIDITY_GRAB,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.9,
            strength=0.9,
            direction="down",
            metadata={},
            volume_anomaly=4.0,
            price_impact=0.08,
            order_book_imbalance=0.9,
            spread_widening=0.06,
            depth_absorption=0.9,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для захвата ликвидности (высокая осторожность)
        assert agent_context.strategy_modifiers.order_aggressiveness < 0.6
        assert agent_context.strategy_modifiers.position_size_multiplier < 0.5
        assert agent_context.strategy_modifiers.confidence_multiplier < 0.7
        assert agent_context.strategy_modifiers.risk_multiplier > 1.5
        assert agent_context.strategy_modifiers.execution_delay_ms > 400

    def test_agent_context_pump_and_dump_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для накачки и сброса."""
        pattern = PatternDetection(
            pattern_type=PatternType.PUMP_AND_DUMP,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.8,
            direction="up",
            metadata={},
            volume_anomaly=5.0,
            price_impact=0.15,
            order_book_imbalance=0.8,
            spread_widening=0.08,
            depth_absorption=0.7,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для накачки и сброса (избегаем)
        assert agent_context.strategy_modifiers.order_aggressiveness < 0.4
        assert agent_context.strategy_modifiers.position_size_multiplier < 0.3
        assert agent_context.strategy_modifiers.confidence_multiplier < 0.5
        assert agent_context.strategy_modifiers.risk_multiplier > 1.8

    def test_agent_context_stop_hunting_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для охоты за стопами."""
        pattern = PatternDetection(
            pattern_type=PatternType.STOP_HUNTING,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.7,
            strength=0.7,
            direction="down",
            metadata={},
            volume_anomaly=2.5,
            price_impact=0.06,
            order_book_imbalance=0.7,
            spread_widening=0.04,
            depth_absorption=0.6,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для охоты за стопами
        assert agent_context.strategy_modifiers.order_aggressiveness < 0.5
        assert agent_context.strategy_modifiers.position_size_multiplier < 0.4
        assert agent_context.strategy_modifiers.confidence_multiplier < 0.6
        assert agent_context.strategy_modifiers.risk_multiplier > 1.6
        assert agent_context.strategy_modifiers.price_offset_percent > 0.0

    def test_agent_context_accumulation_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для накопления."""
        pattern = PatternDetection(
            pattern_type=PatternType.ACCUMULATION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.7,
            direction="up",
            metadata={},
            volume_anomaly=1.5,
            price_impact=0.02,
            order_book_imbalance=0.4,
            spread_widening=0.01,
            depth_absorption=0.5,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для накопления (умеренная агрессивность)
        assert agent_context.strategy_modifiers.order_aggressiveness > 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier > 1.0
        assert agent_context.strategy_modifiers.confidence_multiplier > 1.0

    def test_agent_context_distribution_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для распределения."""
        pattern = PatternDetection(
            pattern_type=PatternType.DISTRIBUTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.7,
            strength=0.6,
            direction="down",
            metadata={},
            volume_anomaly=1.8,
            price_impact=0.03,
            order_book_imbalance=0.6,
            spread_widening=0.02,
            depth_absorption=0.5,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем модификаторы для распределения (снижаем активность)
        assert agent_context.strategy_modifiers.order_aggressiveness < 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier < 1.0
        assert agent_context.strategy_modifiers.confidence_multiplier < 1.0

    def test_agent_context_high_confidence_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для высокой уверенности."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.95,  # Высокая уверенность
            strength=0.9,
            direction="up",
            metadata={},
            volume_anomaly=2.0,
            price_impact=0.04,
            order_book_imbalance=0.7,
            spread_widening=0.02,
            depth_absorption=0.6,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что высокая уверенность усиливает модификаторы
        assert agent_context.strategy_modifiers.order_aggressiveness > 1.1
        assert agent_context.strategy_modifiers.position_size_multiplier > 1.05
        assert agent_context.strategy_modifiers.confidence_multiplier > 1.1

    def test_agent_context_low_confidence_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для низкой уверенности."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.3,  # Низкая уверенность
            strength=0.4,
            direction="up",
            metadata={},
            volume_anomaly=1.2,
            price_impact=0.01,
            order_book_imbalance=0.3,
            spread_widening=0.005,
            depth_absorption=0.3,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что низкая уверенность ослабляет модификаторы
        assert agent_context.strategy_modifiers.order_aggressiveness < 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier < 1.0
        assert agent_context.strategy_modifiers.confidence_multiplier < 1.0

    def test_agent_context_strong_pattern_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для сильного паттерна."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.95,  # Сильный паттерн
            direction="up",
            metadata={},
            volume_anomaly=3.0,
            price_impact=0.06,
            order_book_imbalance=0.8,
            spread_widening=0.03,
            depth_absorption=0.8,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что сильный паттерн усиливает модификаторы
        assert agent_context.strategy_modifiers.confidence_multiplier > 1.1

    def test_agent_context_weak_pattern_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для слабого паттерна."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.2,  # Слабый паттерн
            direction="up",
            metadata={},
            volume_anomaly=1.1,
            price_impact=0.005,
            order_book_imbalance=0.2,
            spread_widening=0.001,
            depth_absorption=0.2,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что слабый паттерн ослабляет модификаторы
        assert agent_context.strategy_modifiers.confidence_multiplier < 1.0

    def test_agent_context_high_volume_anomaly_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для аномально высокого объема."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.7,
            direction="up",
            metadata={},
            volume_anomaly=3.5,  # Аномально высокий объем
            price_impact=0.05,
            order_book_imbalance=0.7,
            spread_widening=0.02,
            depth_absorption=0.6,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что аномально высокий объем увеличивает риск
        assert agent_context.strategy_modifiers.risk_multiplier > 1.0

    def test_agent_context_high_price_impact_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для высокого влияния на цену."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.7,
            direction="up",
            metadata={},
            volume_anomaly=2.0,
            price_impact=0.08,  # Высокое влияние на цену
            order_book_imbalance=0.7,
            spread_widening=0.02,
            depth_absorption=0.6,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что высокое влияние на цену увеличивает смещение цены
        assert agent_context.strategy_modifiers.price_offset_percent > 0.0

    def test_agent_context_high_orderbook_imbalance_modifier(self, agent_context, market_pattern_recognizer) -> None:
        """Тест модификатора для сильного дисбаланса стакана."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.8,
            strength=0.7,
            direction="up",
            metadata={},
            volume_anomaly=2.0,
            price_impact=0.04,
            order_book_imbalance=0.85,  # Сильный дисбаланс стакана
            spread_widening=0.02,
            depth_absorption=0.6,
        )
        agent_context.market_pattern_result = pattern
        agent_context.apply_market_pattern_modifier()
        # Проверяем, что сильный дисбаланс стакана увеличивает задержку исполнения
        assert agent_context.strategy_modifiers.execution_delay_ms > 0

    def test_agent_context_get_market_pattern_status(self, agent_context, market_pattern_recognizer) -> None:
        """Тест получения статуса распознавания паттернов."""
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.85,
            strength=0.8,
            direction="up",
            metadata={},
            volume_anomaly=2.5,
            price_impact=0.03,
            order_book_imbalance=0.6,
            spread_widening=0.02,
            depth_absorption=0.7,
        )
        agent_context.market_pattern_result = pattern
        status = agent_context.get_market_pattern_status()
        # Проверяем статус
        assert status["pattern_detected"] is True
        assert status["pattern_type"] == "whale_absorption"
        assert status["confidence"] == 0.85
        assert status["strength"] == 0.8
        assert status["direction"] == "up"
        assert status["volume_anomaly"] == 2.5
        assert status["price_impact"] == 0.03
        assert status["order_book_imbalance"] == 0.6
        assert status["spread_widening"] == 0.02
        assert status["depth_absorption"] == 0.7
        assert status["status"] == "high_risk"

    def test_agent_context_get_market_pattern_status_no_pattern(self, agent_context, market_pattern_recognizer) -> None:
        """Тест получения статуса без паттерна."""
        status = agent_context.get_market_pattern_status()
        # Проверяем статус без паттерна
        assert status["pattern_detected"] is False
        assert status["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_trading_orchestrator_integration(self, trading_orchestrator) -> None:
        """Тест интеграции в TradingOrchestrator."""
        # Проверяем, что MarketPatternRecognizer добавлен в оркестратор
        assert trading_orchestrator.market_pattern_recognizer is not None
        assert isinstance(trading_orchestrator.market_pattern_recognizer, MarketPatternRecognizer)

    @pytest.mark.asyncio
    async def test_trading_orchestrator_update_market_pattern_analysis(self, trading_orchestrator) -> None:
        """Тест обновления анализа паттернов в оркестраторе."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        # Мокаем методы получения данных
        trading_orchestrator._get_market_data_for_pattern_analysis = AsyncMock(return_value=Mock())
        trading_orchestrator._get_orderbook_data_for_pattern_analysis = AsyncMock(return_value=Mock())
        # Мокаем методы распознавания паттернов
        trading_orchestrator.market_pattern_recognizer.detect_whale_absorption = Mock(return_value=None)
        trading_orchestrator.market_pattern_recognizer.detect_mm_spoofing = Mock(return_value=None)
        trading_orchestrator.market_pattern_recognizer.detect_iceberg_detection = Mock(return_value=None)
        # Вызываем метод обновления
        await trading_orchestrator._update_market_pattern_analysis(symbols)
        # Проверяем, что методы были вызваны
        assert trading_orchestrator._get_market_data_for_pattern_analysis.call_count == 2
        assert trading_orchestrator._get_orderbook_data_for_pattern_analysis.call_count == 2

    @pytest.mark.asyncio
    async def test_trading_orchestrator_apply_market_pattern_analysis(self, trading_orchestrator) -> None:
        """Тест применения анализа паттернов в оркестраторе."""
        # Создаем тестовый сигнал
        signal = Mock()
        signal.confidence = 0.8
        signal.strength = Decimal("1.0")
        signal.metadata = {}
        # Создаем тестовый паттерн
        pattern = PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol="BTCUSDT",
            timestamp=Mock(),
            confidence=0.85,
            strength=0.8,
            direction="up",
            metadata={},
            volume_anomaly=2.5,
            price_impact=0.03,
            order_book_imbalance=0.6,
            spread_widening=0.02,
            depth_absorption=0.7,
        )
        # Устанавливаем паттерн в кэш
        trading_orchestrator._pattern_recognition_cache["BTCUSDT"] = {
            "pattern": pattern,
            "all_patterns": [pattern],
            "timestamp": 1234567890,
        }
        # Применяем анализ
        modified_signal = await trading_orchestrator._apply_market_pattern_analysis("BTCUSDT", signal)
        # Проверяем, что сигнал был модифицирован
        assert modified_signal.confidence > 0.8
        assert modified_signal.strength > Decimal("1.0")

    def test_di_container_trading_orchestrator_creation(self: "TestMarketPatternRecognizerIntegration") -> None:
        """Тест создания TradingOrchestrator через DI контейнер."""
        config = ContainerConfig(
            market_pattern_recognition_enabled=True, mirror_mapping_enabled=True, noise_analysis_enabled=True
        )
        container = DIContainer(config)
        # Создаем TradingOrchestrator
        orchestrator = container.get("trading_orchestrator_use_case")
        # Проверяем, что MarketPatternRecognizer добавлен
        assert orchestrator.market_pattern_recognizer is not None
        assert isinstance(orchestrator.market_pattern_recognizer, MarketPatternRecognizer)

    def test_di_container_trading_orchestrator_creation_disabled(
        self: "TestMarketPatternRecognizerIntegration",
    ) -> None:
        """Тест создания TradingOrchestrator без MarketPatternRecognizer."""
        config = ContainerConfig(market_pattern_recognition_enabled=False)
        container = DIContainer(config)
        # Создаем TradingOrchestrator
        orchestrator = container.get("trading_orchestrator_use_case")
        # Проверяем, что MarketPatternRecognizer не добавлен
        assert orchestrator.market_pattern_recognizer is None
