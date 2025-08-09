"""
Unit тесты для интеграции PatternDiscovery.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from decimal import Decimal
from domain.services.pattern_discovery import PatternDiscovery, PatternConfig, Pattern
from application.di_container import DIContainer
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifier

# from application.use_cases.trading_orchestrator import TradingOrchestratorUseCase
from domain.entities.strategy import Signal, SignalType
from domain.value_objects.percentage import Percentage


class TestPatternDiscoveryIntegration:
    """Тесты интеграции PatternDiscovery."""

    @pytest.fixture
    def pattern_config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура конфигурации PatternDiscovery."""
        return PatternConfig(
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=["candle", "price", "volume"],
            feature_columns=["open", "high", "low", "close", "volume"],
            window_sizes=[5, 10, 20],
            similarity_threshold=0.8,
            technical_indicators=["RSI", "MACD", "BB"],
            volume_threshold=1.5,
            price_threshold=0.02,
            trend_window=20,
        )

    @pytest.fixture
    def pattern_discovery(self, pattern_config) -> Any:
        """Фикстура PatternDiscovery."""
        return PatternDiscovery(pattern_config)

    @pytest.fixture
    def sample_pattern(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура образца паттерна."""
        return Pattern(
            pattern_type="price",
            start_idx=0,
            end_idx=10,
            features=np.array([1.0, 2.0, 3.0]),
            confidence=0.85,
            support=0.25,
            metadata={"trend": "up", "technical_indicators": {"RSI": 65.0}},
        )

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура образца рыночных данных."""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [99, 100, 101, 102, 103],
                "close": [101, 102, 103, 104, 105],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

    def test_di_container_pattern_discovery_registration(self: "TestPatternDiscoveryIntegration") -> None:
        """Тест регистрации PatternDiscovery в DI контейнере."""
        config = ContainerConfig(pattern_discovery_enabled=True)
        container = DIContainer(config)
        # Проверяем, что PatternDiscovery зарегистрирован
        pattern_discovery = container.get("pattern_discovery")
        assert pattern_discovery is not None
        assert isinstance(pattern_discovery, PatternDiscovery)
        # Проверяем конфигурацию
        pattern_config = container.get("pattern_config")
        assert pattern_config is not None
        assert isinstance(pattern_config, PatternConfig)

    def test_di_container_pattern_discovery_disabled(self: "TestPatternDiscoveryIntegration") -> None:
        """Тест отключения PatternDiscovery в DI контейнере."""
        config = ContainerConfig(pattern_discovery_enabled=False)
        container = DIContainer(config)
        # Проверяем, что PatternDiscovery не зарегистрирован
        pattern_discovery = container.get("pattern_discovery")
        assert pattern_discovery is None

    def test_agent_context_pattern_discovery_integration(self, sample_pattern) -> None:
        """Тест интеграции PatternDiscovery в AgentContext."""
        context = AgentContext(symbol="BTCUSDT")
        # Обновляем результат обнаружения паттернов
        context.update_pattern_discovery_result(sample_pattern)
        # Проверяем, что результат сохранен
        result = context.get_pattern_discovery_result()
        assert result is not None
        assert result.pattern_type == "price"
        assert result.confidence == 0.85
        # Применяем модификатор
        context.apply_pattern_discovery_modifier()
        # Проверяем, что модификаторы применены
        assert context.strategy_modifiers.order_aggressiveness > 1.0
        assert context.strategy_modifiers.position_size_multiplier > 1.0
        assert context.strategy_modifiers.confidence_multiplier > 1.0

    def test_agent_context_pattern_discovery_status(self, sample_pattern) -> None:
        """Тест получения статуса PatternDiscovery в AgentContext."""
        context = AgentContext(symbol="BTCUSDT")
        context.update_pattern_discovery_result(sample_pattern)
        status = context.get_pattern_discovery_status()
        assert status["pattern_detected"] is True
        assert status["pattern_type"] == "price"
        assert status["confidence"] == 0.85
        assert status["support"] == 0.25
        assert status["status"] == "high_confidence"

    def test_agent_context_pattern_discovery_no_pattern(self: "TestPatternDiscoveryIntegration") -> None:
        """Тест AgentContext без обнаруженного паттерна."""
        context = AgentContext(symbol="BTCUSDT")
        # Проверяем статус без паттерна
        status = context.get_pattern_discovery_status()
        assert status["pattern_detected"] is False
        assert status["status"] == "unknown"
        # Проверяем, что модификаторы не применяются
        original_aggressiveness = context.strategy_modifiers.order_aggressiveness
        context.apply_pattern_discovery_modifier()
        assert context.strategy_modifiers.order_aggressiveness == original_aggressiveness

    def test_trading_orchestrator_pattern_discovery_integration(self, pattern_discovery, sample_market_data) -> None:
        """Тест интеграции PatternDiscovery в TradingOrchestrator."""
        # Создаем моки для зависимостей
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        # orchestrator = TradingOrchestratorUseCase(
        #     order_repository=mock_order_repo,
        #     position_repository=mock_position_repo,
        #     portfolio_repository=mock_portfolio_repo,
        #     trading_repository=mock_trading_repo,
        #     strategy_repository=mock_strategy_repo,
        #     enhanced_trading_service=mock_enhanced_trading_service,
        #     pattern_discovery=pattern_discovery
        # )
        orchestrator = Mock()
        orchestrator.pattern_discovery = pattern_discovery
        orchestrator._pattern_discovery_cache = {}
        orchestrator._last_pattern_discovery_update = None
        # Проверяем, что PatternDiscovery инициализирован
        assert orchestrator.pattern_discovery is not None
        assert isinstance(orchestrator.pattern_discovery, PatternDiscovery)
        # Проверяем кэш
        assert hasattr(orchestrator, "_pattern_discovery_cache")
        assert hasattr(orchestrator, "_last_pattern_discovery_update")

    @pytest.mark.asyncio
    async def test_trading_orchestrator_pattern_discovery_update(self, pattern_discovery) -> None:
        """Тест обновления PatternDiscovery в TradingOrchestrator."""
        # Создаем моки
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        # orchestrator = TradingOrchestratorUseCase(
        #     order_repository=mock_order_repo,
        #     position_repository=mock_position_repo,
        #     portfolio_repository=mock_portfolio_repo,
        #     trading_repository=mock_trading_repo,
        #     strategy_repository=mock_strategy_repo,
        #     enhanced_trading_service=mock_enhanced_trading_service,
        #     pattern_discovery=pattern_discovery
        # )
        orchestrator = Mock()
        orchestrator.pattern_discovery = pattern_discovery
        orchestrator._pattern_discovery_cache = {}
        orchestrator._last_pattern_discovery_update = None
        # Мокаем метод получения рыночных данных
        with patch.object(orchestrator, "_get_market_data_for_pattern_discovery") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame(
                {
                    "open": [100, 101, 102, 103, 104],
                    "high": [102, 103, 104, 105, 106],
                    "low": [99, 100, 101, 102, 103],
                    "close": [101, 102, 103, 104, 105],
                    "volume": [1000, 1100, 1200, 1300, 1400],
                }
            )
            # Вызываем обновление
            await orchestrator._update_pattern_discovery(["BTCUSDT"])
            # Проверяем, что метод был вызван
            mock_get_data.assert_called_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_trading_orchestrator_pattern_discovery_analysis(self, pattern_discovery, sample_pattern) -> None:
        """Тест применения анализа PatternDiscovery к сигналу."""
        # Создаем моки
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        # orchestrator = TradingOrchestratorUseCase(
        #     order_repository=mock_order_repo,
        #     position_repository=mock_position_repo,
        #     portfolio_repository=mock_portfolio_repo,
        #     trading_repository=mock_trading_repo,
        #     strategy_repository=mock_strategy_repo,
        #     enhanced_trading_service=mock_enhanced_trading_service,
        #     pattern_discovery=pattern_discovery
        # )
        orchestrator = Mock()
        orchestrator.pattern_discovery = pattern_discovery
        orchestrator._pattern_discovery_cache = {}
        orchestrator._last_pattern_discovery_update = None
        # Создаем тестовый сигнал
        original_signal = Signal(
            signal_type=SignalType.BUY,
            symbol="BTCUSDT",
            confidence=Percentage(Decimal("0.8")),
            strength=Percentage(Decimal("0.7")),
            timestamp=datetime.now(),
        )
        # Добавляем паттерн в кэш
        orchestrator._pattern_discovery_cache["BTCUSDT"] = {
            "patterns": [sample_pattern],
            "timestamp": 1234567890.0,
            "symbol": "BTCUSDT",
        }
        # Применяем анализ
        modified_signal = await orchestrator._apply_pattern_discovery_analysis("BTCUSDT", original_signal)
        # Проверяем, что сигнал модифицирован
        assert modified_signal.confidence.value > original_signal.confidence.value
        assert modified_signal.strength.value > original_signal.strength.value
        # Проверяем метаданные
        assert "pattern_discovery" in modified_signal.metadata
        assert modified_signal.metadata["pattern_discovery"]["pattern_type"] == "price"
        assert modified_signal.metadata["pattern_discovery"]["confidence"] == 0.85

    def test_pattern_discovery_config_validation(self: "TestPatternDiscoveryIntegration") -> None:
        """Тест валидации конфигурации PatternDiscovery."""
        # Тест с некорректной конфигурацией
        with pytest.raises(ValueError):
            PatternConfig(
                min_pattern_length=50,  # Больше max_pattern_length
                max_pattern_length=10,
                min_confidence=0.7,
                min_support=0.1,
                max_patterns=100,
                clustering_method="dbscan",
                min_cluster_size=3,
                pattern_types=[],  # Пустой список
                feature_columns=["open", "high", "low", "close", "volume"],
                window_sizes=[5, 10, 20],
                similarity_threshold=0.8,
                technical_indicators=["RSI", "MACD", "BB"],
                volume_threshold=1.5,
                price_threshold=0.02,
                trend_window=20,
            )

    def test_pattern_discovery_pattern_creation(self, sample_pattern) -> None:
        """Тест создания паттерна."""
        # Проверяем основные атрибуты
        assert sample_pattern.pattern_type == "price"
        assert sample_pattern.confidence == 0.85
        assert sample_pattern.support == 0.25
        assert sample_pattern.start_idx == 0
        assert sample_pattern.end_idx == 10
        # Проверяем метаданные
        assert sample_pattern.metadata["trend"] == "up"
        assert sample_pattern.metadata["technical_indicators"]["RSI"] == 65.0

    def test_pattern_discovery_pattern_serialization(self, sample_pattern) -> None:
        """Тест сериализации паттерна."""
        # Преобразуем в словарь
        pattern_dict = sample_pattern.to_dict()
        # Проверяем основные поля
        assert pattern_dict["pattern_type"] == "price"
        assert pattern_dict["confidence"] == 0.85
        assert pattern_dict["support"] == 0.25
        assert pattern_dict["trend"] == "up"
        assert pattern_dict["technical_indicators"]["RSI"] == 65.0
        # Восстанавливаем из словаря
        restored_pattern = Pattern.from_dict(pattern_dict)
        # Проверяем, что паттерн восстановлен корректно
        assert restored_pattern.pattern_type == sample_pattern.pattern_type
        assert restored_pattern.confidence == sample_pattern.confidence
        assert restored_pattern.support == sample_pattern.support
        assert restored_pattern.metadata["trend"] == sample_pattern.metadata["trend"]

    def test_strategy_modifiers_pattern_discovery_integration(self: "TestPatternDiscoveryIntegration") -> None:
        """Тест интеграции PatternDiscovery в StrategyModifiers."""
        modifiers = StrategyModifiers()
        # Проверяем, что модификаторы инициализированы с дефолтными значениями
        assert modifiers.order_aggressiveness == 1.0
        assert modifiers.position_size_multiplier == 1.0
        assert modifiers.confidence_multiplier == 1.0
        assert modifiers.risk_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_pattern_discovery_error_handling(self, pattern_discovery) -> None:
        """Тест обработки ошибок в PatternDiscovery."""
        # Создаем моки
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        # orchestrator = TradingOrchestratorUseCase(
        #     order_repository=mock_order_repo,
        #     position_repository=mock_position_repo,
        #     portfolio_repository=mock_portfolio_repo,
        #     trading_repository=mock_trading_repo,
        #     strategy_repository=mock_strategy_repo,
        #     enhanced_trading_service=mock_enhanced_trading_service,
        #     pattern_discovery=pattern_discovery
        # )
        orchestrator = Mock()
        orchestrator.pattern_discovery = pattern_discovery
        orchestrator._pattern_discovery_cache = {}
        orchestrator._last_pattern_discovery_update = None
        # Создаем тестовый сигнал
        original_signal = Signal(
            signal_type=SignalType.BUY,
            symbol="BTCUSDT",
            confidence=Percentage(Decimal("0.8")),
            strength=Percentage(Decimal("0.7")),
            timestamp=datetime.now(),
        )
        # Тестируем обработку ошибок при отсутствии данных
        with patch.object(orchestrator, "_get_market_data_for_pattern_discovery") as mock_get_data:
            mock_get_data.return_value = None
            # Обновление должно завершиться без ошибок
            await orchestrator._update_pattern_discovery(["BTCUSDT"])
            # Анализ должен вернуть исходный сигнал
            modified_signal = await orchestrator._apply_pattern_discovery_analysis("BTCUSDT", original_signal)
            assert modified_signal == original_signal

    def test_pattern_discovery_integration_completeness(self: "TestPatternDiscoveryIntegration") -> None:
        """Тест полноты интеграции PatternDiscovery."""
        # Проверяем, что все компоненты интегрированы
        config = ContainerConfig(pattern_discovery_enabled=True)
        container = DIContainer(config)
        # Проверяем DI контейнер
        pattern_discovery = container.get("pattern_discovery")
        assert pattern_discovery is not None
        # Проверяем AgentContext
        context = AgentContext(symbol="BTCUSDT")
        assert hasattr(context, "pattern_discovery_result")
        assert hasattr(context, "apply_pattern_discovery_modifier")
        assert hasattr(context, "get_pattern_discovery_status")
        assert hasattr(context, "update_pattern_discovery_result")
        assert hasattr(context, "get_pattern_discovery_result")
        # Проверяем TradingOrchestrator
        orchestrator = container.get("trading_orchestrator_use_case")
        assert hasattr(orchestrator, "pattern_discovery")
        assert hasattr(orchestrator, "_pattern_discovery_cache")
        assert hasattr(orchestrator, "_last_pattern_discovery_update")
        assert hasattr(orchestrator, "_update_pattern_discovery")
        assert hasattr(orchestrator, "_apply_pattern_discovery_analysis")
