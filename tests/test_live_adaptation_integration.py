"""
Unit тесты для интеграции LiveAdaptationModel в систему Syntra.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, MagicMock
from decimal import Decimal
from infrastructure.ml_services.live_adaptation import LiveAdaptation
from domain.value_objects.signal import Signal, SignalType
from domain.value_objects.money import Money
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifier
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
from application.di_container import Container
import pandas as pd
from shared.numpy_utils import np


class TestLiveAdaptationIntegration:
    """Тесты интеграции LiveAdaptationModel в систему Syntra."""

    @pytest.fixture
    def mock_live_adaptation_model(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок для LiveAdaptationModel."""
        mock = Mock(spec=LiveAdaptation)
        # Создаем мок метрик адаптации
        mock_metrics = Mock()
        mock_metrics.confidence = 0.85
        mock_metrics.drift_score = 0.3
        mock_metrics.accuracy = 0.78
        mock_metrics.precision = 0.82
        mock_metrics.recall = 0.75
        mock_metrics.f1 = 0.78
        mock.update = AsyncMock()
        mock.get_metrics = AsyncMock(return_value=mock_metrics)
        mock.predict = AsyncMock(return_value=(1.0, 0.85))
        return mock

    @pytest.fixture
    def mock_agent_context(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок для AgentContext."""
        context = AgentContext(symbol="BTC/USD")
        context.live_adaptation_result = None
        return context

    @pytest.fixture
    def mock_trading_orchestrator(self, mock_live_adaptation_model) -> Any:
        """Мок для TradingOrchestrator с LiveAdaptationModel."""
        orchestrator = Mock(spec=DefaultTradingOrchestratorUseCase)
        orchestrator.live_adaptation_model = mock_live_adaptation_model
        orchestrator._live_adaptation_cache = {}
        orchestrator._last_live_adaptation_update = None
        return orchestrator

    @pytest.fixture
    def sample_signal(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Образец торгового сигнала."""
        return Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            price=Money(Decimal("50000"), Currency.USD),
            volume=Volume(Decimal("0.1")),
            confidence=Percentage(Decimal("0.8")),
            strength=Percentage(Decimal("0.7")),
            timestamp=1234567890,
            metadata={},
        )

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Образец рыночных данных."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-02", freq="1H")
        data = {
            "open": np.random.uniform(45000, 55000, len(dates)),
            "high": np.random.uniform(45000, 55000, len(dates)),
            "low": np.random.uniform(45000, 55000, len(dates)),
            "close": np.random.uniform(45000, 55000, len(dates)),
            "volume": np.random.uniform(100, 1000, len(dates)),
        }
        return pd.DataFrame(data, index=dates)

    def test_di_container_integration(self: "TestLiveAdaptationIntegration") -> None:
        """Тест интеграции LiveAdaptationModel в DI контейнер."""
        # Создаем контейнер
        container = Container()
        # Проверяем, что LiveAdaptationModel зарегистрирован
        live_adaptation_model = container.live_adaptation_model()
        assert live_adaptation_model is not None
        assert isinstance(live_adaptation_model, LiveAdaptation)

    def test_agent_context_live_adaptation_field(self, mock_agent_context) -> None:
        """Тест поля live_adaptation_result в AgentContext."""
        # Проверяем, что поле существует
        assert hasattr(mock_agent_context, "live_adaptation_result")
        assert mock_agent_context.live_adaptation_result is None

    def test_agent_context_apply_live_adaptation_modifier(self, mock_agent_context, sample_signal) -> None:
        """Тест метода apply_live_adaptation_modifier в AgentContext."""
        # Создаем мок результат адаптации
        adaptation_data = {
            "confidence": 0.85,
            "drift_score": 0.3,
            "accuracy": 0.78,
            "precision": 0.82,
            "recall": 0.75,
            "f1": 0.78,
        }
        mock_agent_context.live_adaptation_result = adaptation_data
        # Применяем модификатор
        modified_signal = mock_agent_context.apply_live_adaptation_modifier(sample_signal)
        # Проверяем, что сигнал был модифицирован
        assert modified_signal is not None
        assert modified_signal.confidence > sample_signal.confidence  # Увеличена уверенность
        assert "live_adaptation" in modified_signal.metadata

    def test_trading_orchestrator_live_adaptation_constructor(self, mock_live_adaptation_model) -> None:
        """Тест конструктора TradingOrchestrator с LiveAdaptationModel."""
        # Создаем моки для зависимостей
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        # Создаем оркестратор
        orchestrator = DefaultTradingOrchestratorUseCase(
            order_repository=mock_order_repo,
            position_repository=mock_position_repo,
            portfolio_repository=mock_portfolio_repo,
            trading_repository=mock_trading_repo,
            strategy_repository=mock_strategy_repo,
            enhanced_trading_service=mock_enhanced_trading_service,
            live_adaptation_model=mock_live_adaptation_model,
        )
        # Проверяем, что LiveAdaptationModel был добавлен
        assert orchestrator.live_adaptation_model == mock_live_adaptation_model
        assert hasattr(orchestrator, "_live_adaptation_cache")
        assert hasattr(orchestrator, "_last_live_adaptation_update")

    @pytest.mark.asyncio
    async def test_update_live_adaptation(self, mock_trading_orchestrator, sample_market_data) -> None:
        """Тест метода _update_live_adaptation."""
        symbols = ["BTC/USD", "ETH/USD"]
        # Мокаем получение рыночных данных
        mock_trading_orchestrator._get_market_data_for_adaptation = AsyncMock(return_value=sample_market_data)
        # Вызываем метод
        await mock_trading_orchestrator._update_live_adaptation(symbols)
        # Проверяем, что LiveAdaptationModel был вызван
        mock_trading_orchestrator.live_adaptation_model.update.assert_called()
        mock_trading_orchestrator.live_adaptation_model.get_metrics.assert_called()
        # Проверяем, что кэш был обновлен
        assert len(mock_trading_orchestrator._live_adaptation_cache) > 0
        assert mock_trading_orchestrator._last_live_adaptation_update is not None

    @pytest.mark.asyncio
    async def test_apply_live_adaptation_analysis(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест метода _apply_live_adaptation_analysis."""
        symbol = "BTC/USD"
        # Подготавливаем кэш
        mock_metrics = Mock()
        mock_metrics.confidence = 0.85
        mock_metrics.drift_score = 0.3
        mock_metrics.accuracy = 0.78
        mock_metrics.precision = 0.82
        mock_metrics.recall = 0.75
        mock_metrics.f1 = 0.78
        mock_trading_orchestrator._live_adaptation_cache[symbol] = {"metrics": mock_metrics, "timestamp": 1234567890}
        # Применяем анализ
        modified_signal = await mock_trading_orchestrator._apply_live_adaptation_analysis(symbol, sample_signal)
        # Проверяем результат
        assert modified_signal is not None
        assert modified_signal.confidence > sample_signal.confidence
        assert "live_adaptation" in modified_signal.metadata

    @pytest.mark.asyncio
    async def test_live_adaptation_in_execute_strategy(self, mock_trading_orchestrator) -> None:
        """Тест интеграции LiveAdaptationModel в execute_strategy."""
        # Создаем мок запрос
        mock_request = Mock()
        mock_request.symbol = "BTC/USD"
        mock_request.risk_level = "medium"
        # Мокаем другие методы
        mock_trading_orchestrator._update_live_adaptation = AsyncMock()
        mock_trading_orchestrator._apply_live_adaptation_analysis = AsyncMock(return_value=sample_signal)
        # Вызываем execute_strategy (частично)
        await mock_trading_orchestrator._update_live_adaptation([mock_request.symbol])
        # Проверяем, что LiveAdaptationModel был обновлен
        mock_trading_orchestrator._update_live_adaptation.assert_called_once_with([mock_request.symbol])

    @pytest.mark.asyncio
    async def test_live_adaptation_in_process_signal(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест интеграции LiveAdaptationModel в process_signal."""
        # Создаем мок запрос
        mock_request = Mock()
        mock_request.signal = sample_signal
        # Мокаем методы
        mock_trading_orchestrator._update_live_adaptation = AsyncMock()
        mock_trading_orchestrator._apply_live_adaptation_analysis = AsyncMock(return_value=sample_signal)
        # Вызываем process_signal (частично)
        await mock_trading_orchestrator._update_live_adaptation([mock_request.signal.trading_pair])
        modified_signal = await mock_trading_orchestrator._apply_live_adaptation_analysis(
            mock_request.signal.trading_pair, mock_request.signal
        )
        # Проверяем, что LiveAdaptationModel был применен
        mock_trading_orchestrator._update_live_adaptation.assert_called_once_with([mock_request.signal.trading_pair])
        mock_trading_orchestrator._apply_live_adaptation_analysis.assert_called_once_with(
            mock_request.signal.trading_pair, mock_request.signal
        )
        assert modified_signal is not None

    def test_live_adaptation_strategy_modifiers(self: "TestLiveAdaptationIntegration") -> None:
        """Тест модификаторов стратегий для LiveAdaptationModel."""
        modifiers = StrategyModifiers()
        # Проверяем, что модификаторы для LiveAdaptationModel существуют
        assert hasattr(modifiers, "live_adaptation_confidence_multiplier")
        assert hasattr(modifiers, "live_adaptation_strength_multiplier")
        assert hasattr(modifiers, "live_adaptation_execution_delay_ms")

    def test_live_adaptation_error_handling(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест обработки ошибок в LiveAdaptationModel."""
        symbol = "BTC/USD"
        # Симулируем ошибку в LiveAdaptationModel
        mock_trading_orchestrator.live_adaptation_model.update.side_effect = Exception("Live adaptation error")
        # Проверяем, что ошибка обрабатывается корректно
        try:
            # Применяем анализ - должен вернуть исходный сигнал
            modified_signal = mock_trading_orchestrator._apply_live_adaptation_analysis(symbol, sample_signal)
            assert modified_signal == sample_signal
        except Exception:
            pytest.fail("Live adaptation error should be handled gracefully")

    def test_live_adaptation_cache_management(self, mock_trading_orchestrator) -> None:
        """Тест управления кэшем LiveAdaptationModel."""
        symbol = "BTC/USD"
        # Проверяем, что кэш пустой изначально
        assert symbol not in mock_trading_orchestrator._live_adaptation_cache
        # Добавляем данные в кэш
        mock_trading_orchestrator._live_adaptation_cache[symbol] = {"metrics": Mock(), "timestamp": 1234567890}
        # Проверяем, что данные добавлены
        assert symbol in mock_trading_orchestrator._live_adaptation_cache
        assert mock_trading_orchestrator._live_adaptation_cache[symbol]["timestamp"] == 1234567890

    def test_live_adaptation_metadata_structure(self, mock_agent_context, sample_signal) -> None:
        """Тест структуры метаданных LiveAdaptationModel."""
        # Создаем мок результат
        adaptation_data = {
            "confidence": 0.85,
            "drift_score": 0.3,
            "accuracy": 0.78,
            "precision": 0.82,
            "recall": 0.75,
            "f1": 0.78,
        }
        mock_agent_context.live_adaptation_result = adaptation_data
        # Применяем модификатор
        modified_signal = mock_agent_context.apply_live_adaptation_modifier(sample_signal)
        # Проверяем структуру метаданных
        assert "live_adaptation" in modified_signal.metadata
        adaptation_metadata = modified_signal.metadata["live_adaptation"]
        assert "confidence" in adaptation_metadata
        assert "drift_score" in adaptation_metadata
        assert "accuracy" in adaptation_metadata
        assert "precision" in adaptation_metadata
        assert "recall" in adaptation_metadata
        assert "f1" in adaptation_metadata
        assert adaptation_metadata["confidence"] == 0.85
        assert adaptation_metadata["drift_score"] == 0.3
        assert adaptation_metadata["accuracy"] == 0.78

    @pytest.mark.asyncio
    async def test_live_adaptation_high_confidence_scenario(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест сценария с высокой уверенностью модели."""
        symbol = "BTC/USD"
        # Создаем метрики с высокой уверенностью
        mock_metrics = Mock()
        mock_metrics.confidence = 0.95
        mock_metrics.drift_score = 0.1
        mock_metrics.accuracy = 0.92
        mock_metrics.precision = 0.94
        mock_metrics.recall = 0.90
        mock_metrics.f1 = 0.92
        mock_trading_orchestrator._live_adaptation_cache[symbol] = {"metrics": mock_metrics, "timestamp": 1234567890}
        # Применяем анализ
        modified_signal = await mock_trading_orchestrator._apply_live_adaptation_analysis(symbol, sample_signal)
        # Проверяем, что сигнал был усилен
        assert modified_signal.confidence > sample_signal.confidence
        assert modified_signal.strength > sample_signal.strength

    @pytest.mark.asyncio
    async def test_live_adaptation_low_confidence_scenario(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест сценария с низкой уверенностью модели."""
        symbol = "BTC/USD"
        # Создаем метрики с низкой уверенностью
        mock_metrics = Mock()
        mock_metrics.confidence = 0.3
        mock_metrics.drift_score = 0.8
        mock_metrics.accuracy = 0.45
        mock_metrics.precision = 0.42
        mock_metrics.recall = 0.48
        mock_metrics.f1 = 0.45
        mock_trading_orchestrator._live_adaptation_cache[symbol] = {"metrics": mock_metrics, "timestamp": 1234567890}
        # Применяем анализ
        modified_signal = await mock_trading_orchestrator._apply_live_adaptation_analysis(symbol, sample_signal)
        # Проверяем, что сигнал был ослаблен
        assert modified_signal.confidence < sample_signal.confidence
        assert modified_signal.strength < sample_signal.strength
        assert "execution_delay_ms" in modified_signal.metadata

    @pytest.mark.asyncio
    async def test_live_adaptation_high_drift_scenario(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест сценария с высоким дрейфом данных."""
        symbol = "BTC/USD"
        # Создаем метрики с высоким дрейфом
        mock_metrics = Mock()
        mock_metrics.confidence = 0.6
        mock_metrics.drift_score = 0.9
        mock_metrics.accuracy = 0.55
        mock_metrics.precision = 0.52
        mock_metrics.recall = 0.58
        mock_metrics.f1 = 0.55
        mock_trading_orchestrator._live_adaptation_cache[symbol] = {"metrics": mock_metrics, "timestamp": 1234567890}
        # Применяем анализ
        modified_signal = await mock_trading_orchestrator._apply_live_adaptation_analysis(symbol, sample_signal)
        # Проверяем, что добавлена задержка исполнения
        assert "execution_delay_ms" in modified_signal.metadata
        assert modified_signal.metadata["execution_delay_ms"] >= 1000

    def test_live_adaptation_performance_metrics(self, mock_agent_context, sample_signal) -> None:
        """Тест метрик производительности LiveAdaptationModel."""
        # Создаем мок результат с различными метриками
        adaptation_data = {
            "confidence": 0.85,
            "drift_score": 0.3,
            "accuracy": 0.78,
            "precision": 0.82,
            "recall": 0.75,
            "f1": 0.78,
        }
        mock_agent_context.live_adaptation_result = adaptation_data
        # Применяем модификатор
        modified_signal = mock_agent_context.apply_live_adaptation_modifier(sample_signal)
        # Проверяем, что метрики корректно применены
        adaptation_metadata = modified_signal.metadata["live_adaptation"]
        # Проверяем, что все метрики присутствуют
        for metric in ["confidence", "drift_score", "accuracy", "precision", "recall", "f1"]:
            assert metric in adaptation_metadata
            assert isinstance(adaptation_metadata[metric], (int, float))
            assert 0 <= adaptation_metadata[metric] <= 1

    @pytest.mark.asyncio
    async def test_live_adaptation_async_operations(self, mock_live_adaptation_model) -> None:
        """Тест асинхронных операций адаптации."""
        # Создаем мок для асинхронных операций
        mock_live_adaptation_model.adapt_to_market_changes.return_value = {
            "adaptation_score": 0.85,
            "confidence": 0.9,
            "recommendations": ["increase_position_size", "adjust_stop_loss"],
        }

        # Тестируем асинхронную адаптацию
        result = await mock_live_adaptation_model.adapt_to_market_changes("BTC/USD")

        assert result is not None
        assert "adaptation_score" in result
        assert "confidence" in result
        assert "recommendations" in result


if __name__ == "__main__":
    pytest.main([__file__])
