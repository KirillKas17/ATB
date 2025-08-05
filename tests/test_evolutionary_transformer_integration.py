"""
Unit тесты для интеграции EvolutionaryTransformer.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from decimal import Decimal
from domain.entities.strategy import Signal, SignalType
from domain.value_objects.percentage import Percentage
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifiers
from infrastructure.ml_services.transformer_predictor import EvolutionaryTransformer
from application.use_cases.trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    ExecuteStrategyRequest,
    ProcessSignalRequest
)
class TestEvolutionaryTransformerIntegration:
    """Тесты интеграции EvolutionaryTransformer."""
    @pytest.fixture
    def evolutionary_transformer(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра EvolutionaryTransformer."""
        return EvolutionaryTransformer(
            input_size=64,
            output_size=1,
            population_size=10
        )
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
            "enhanced_trading_service": Mock()
        }
    @pytest.fixture
    def trading_orchestrator(self, mock_repositories, evolutionary_transformer) -> Any:
        """Создание TradingOrchestrator с EvolutionaryTransformer."""
        return DefaultTradingOrchestratorUseCase(
            order_repository=mock_repositories["order_repository"],
            position_repository=mock_repositories["position_repository"],
            portfolio_repository=mock_repositories["portfolio_repository"],
            trading_repository=mock_repositories["trading_repository"],
            strategy_repository=mock_repositories["strategy_repository"],
            enhanced_trading_service=mock_repositories["enhanced_trading_service"],
            evolutionary_transformer=evolutionary_transformer
        )
    def test_evolutionary_transformer_di_integration(self: "TestEvolutionaryTransformerIntegration") -> None:
        """Тест интеграции EvolutionaryTransformer в DI контейнер."""
        from application.di_container import get_container, ContainerConfig
        # Создаем контейнер с включенным EvolutionaryTransformer
        config = ContainerConfig(evolutionary_transformer_enabled=True)
        container = get_container(config)
        # Проверяем, что EvolutionaryTransformer зарегистрирован
        transformer = container.get("evolutionary_transformer")
        assert transformer is not None
        assert isinstance(transformer, EvolutionaryTransformer)
    def test_agent_context_evolutionary_transformer_field(self, agent_context) -> None:
        """Тест поля evolutionary_transformer_result в AgentContext."""
        # Проверяем, что поле существует и изначально None
        assert hasattr(agent_context, 'evolutionary_transformer_result')
        assert agent_context.evolutionary_transformer_result is None
        # Устанавливаем результат
        result = {
            "evolution_score": 0.85,
            "fitness_score": 0.92,
            "adaptation_rate": 1.2,
            "generation": 25,
            "best_model_confidence": 0.95,
            "evolution_state": "converging"
        }
        agent_context.evolutionary_transformer_result = result
        # Проверяем, что результат установлен
        assert agent_context.evolutionary_transformer_result == result
    def test_apply_evolutionary_transformer_modifier(self, agent_context) -> None:
        """Тест применения модификатора эволюционного трансформера."""
        # Устанавливаем результат эволюции
        result = {
            "evolution_score": 0.85,
            "fitness_score": 0.92,
            "adaptation_rate": 1.2,
            "generation": 25,
            "best_model_confidence": 0.95,
            "evolution_state": "converging"
        }
        agent_context.evolutionary_transformer_result = result
        # Запоминаем исходные значения модификаторов
        original_aggressiveness = agent_context.strategy_modifiers.order_aggressiveness
        original_position_size = agent_context.strategy_modifiers.position_size_multiplier
        original_confidence = agent_context.strategy_modifiers.confidence_multiplier
        # Применяем модификатор
        agent_context.apply_evolutionary_transformer_modifier()
        # Проверяем, что модификаторы изменились
        assert agent_context.strategy_modifiers.order_aggressiveness > original_aggressiveness
        assert agent_context.strategy_modifiers.position_size_multiplier > original_position_size
        assert agent_context.strategy_modifiers.confidence_multiplier > original_confidence
    def test_evolutionary_transformer_modifier_with_low_scores(self, agent_context) -> None:
        """Тест модификатора с низкими скорами."""
        # Устанавливаем низкие скора
        result = {
            "evolution_score": 0.2,
            "fitness_score": 0.3,
            "adaptation_rate": 0.4,
            "generation": 5,
            "best_model_confidence": 0.4,
            "evolution_state": "exploring"
        }
        agent_context.evolutionary_transformer_result = result
        # Запоминаем исходные значения
        original_aggressiveness = agent_context.strategy_modifiers.order_aggressiveness
        original_position_size = agent_context.strategy_modifiers.position_size_multiplier
        original_confidence = agent_context.strategy_modifiers.confidence_multiplier
        # Применяем модификатор
        agent_context.apply_evolutionary_transformer_modifier()
        # Проверяем, что модификаторы снизились
        assert agent_context.strategy_modifiers.order_aggressiveness < original_aggressiveness
        assert agent_context.strategy_modifiers.position_size_multiplier < original_position_size
        assert agent_context.strategy_modifiers.confidence_multiplier < original_confidence
    def test_get_evolutionary_transformer_status(self, agent_context) -> None:
        """Тест получения статуса эволюционного трансформера."""
        # Без результата
        status = agent_context.get_evolutionary_transformer_status()
        assert status["evolution_active"] is False
        assert status["status"] == "unknown"
        # С результатом
        result = {
            "evolution_score": 0.85,
            "fitness_score": 0.92,
            "adaptation_rate": 1.2,
            "generation": 25,
            "best_model_confidence": 0.95,
            "evolution_state": "converging"
        }
        agent_context.evolutionary_transformer_result = result
        status = agent_context.get_evolutionary_transformer_status()
        assert status["evolution_active"] is True
        assert status["evolution_score"] == 0.85
        assert status["fitness_score"] == 0.92
        assert status["generation"] == 25
        assert status["status"] == "high_evolution"
    def test_update_evolutionary_transformer_result(self, agent_context) -> None:
        """Тест обновления результата эволюционного трансформера."""
        result = {
            "evolution_score": 0.75,
            "fitness_score": 0.88,
            "adaptation_rate": 1.1,
            "generation": 15,
            "best_model_confidence": 0.87,
            "evolution_state": "stable"
        }
        agent_context.update_evolutionary_transformer_result(result)
        assert agent_context.evolutionary_transformer_result == result
    def test_get_evolutionary_transformer_result(self, agent_context) -> None:
        """Тест получения результата эволюционного трансформера."""
        # Изначально None
        assert agent_context.get_evolutionary_transformer_result() is None
        # После установки
        result = {"evolution_score": 0.8}
        agent_context.evolutionary_transformer_result = result
        assert agent_context.get_evolutionary_transformer_result() == result
    @pytest.mark.asyncio
    async def test_trading_orchestrator_evolutionary_transformer_integration(self, trading_orchestrator) -> None:
        """Тест интеграции EvolutionaryTransformer в TradingOrchestrator."""
        # Проверяем, что EvolutionaryTransformer установлен
        assert trading_orchestrator.evolutionary_transformer is not None
        assert isinstance(trading_orchestrator.evolutionary_transformer, EvolutionaryTransformer)
    @pytest.mark.asyncio
    async def test_update_evolutionary_transformer(self, trading_orchestrator) -> None:
        """Тест обновления эволюционного трансформера."""
        symbols = ["BTCUSDT"]
        # Мокаем получение рыночных данных
        with patch.object(trading_orchestrator, '_get_market_data_for_evolutionary_transformer') as mock_get_data:
            mock_get_data.return_value = Mock()  # Возвращаем мок DataFrame
            # Мокаем метод evolve
            with patch.object(trading_orchestrator.evolutionary_transformer, 'evolve') as mock_evolve:
                mock_evolve.return_value = Mock()
                await trading_orchestrator._update_evolutionary_transformer(symbols)
                # Проверяем, что методы были вызваны
                mock_get_data.assert_called_once_with("BTCUSDT")
                mock_evolve.assert_called_once_with(generations=5)
    @pytest.mark.asyncio
    async def test_apply_evolutionary_transformer_analysis(self, trading_orchestrator) -> None:
        """Тест применения анализа эволюционного трансформера к сигналу."""
        # Создаем тестовый сигнал
        signal = Signal(
            id="test_signal",
            trading_pair="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.7")),
            strength=Percentage(Decimal("0.8")),
            timestamp=datetime.now()
        )
        # Устанавливаем кэш с результатом эволюции
        evolution_result = Mock()
        evolution_result.best_fitness = 0.85
        evolution_result.generation = 20
        evolution_result.population_size = 10
        trading_orchestrator._evolutionary_transformer_cache["BTCUSDT_evolutionary_transformer"] = evolution_result
        # Применяем анализ
        modified_signal = await trading_orchestrator._apply_evolutionary_transformer_analysis("BTCUSDT", signal)
        # Проверяем, что сигнал был модифицирован
        assert modified_signal.confidence.value > signal.confidence.value
        assert modified_signal.strength.value > signal.strength.value
        assert "evolutionary_transformer" in modified_signal.metadata
    @pytest.mark.asyncio
    async def test_apply_evolutionary_transformer_analysis_low_fitness(self, trading_orchestrator) -> None:
        """Тест применения анализа с низкой пригодностью."""
        # Создаем тестовый сигнал
        signal = Signal(
            id="test_signal",
            trading_pair="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.7")),
            strength=Percentage(Decimal("0.8")),
            timestamp=datetime.now()
        )
        # Устанавливаем кэш с низкой пригодностью
        evolution_result = Mock()
        evolution_result.best_fitness = 0.3
        evolution_result.generation = 5
        evolution_result.population_size = 10
        trading_orchestrator._evolutionary_transformer_cache["BTCUSDT_evolutionary_transformer"] = evolution_result
        # Применяем анализ
        modified_signal = await trading_orchestrator._apply_evolutionary_transformer_analysis("BTCUSDT", signal)
        # Проверяем, что сигнал был ослаблен
        assert modified_signal.confidence.value < signal.confidence.value
        assert modified_signal.strength.value < signal.strength.value
    @pytest.mark.asyncio
    async def test_get_market_data_for_evolutionary_transformer(self, trading_orchestrator) -> None:
        """Тест получения рыночных данных для эволюционного трансформера."""
        market_data = await trading_orchestrator._get_market_data_for_evolutionary_transformer("BTCUSDT")
        # Проверяем, что данные получены
        assert market_data is not None
        assert hasattr(market_data, 'columns')
        assert 'timestamp' in market_data.columns
        assert 'close' in market_data.columns
        assert 'volume' in market_data.columns
    def test_strategy_modifiers_evolutionary_transformer_fields(self: "TestEvolutionaryTransformerIntegration") -> None:
        """Тест полей модификаторов для эволюционного трансформера."""
        modifiers = StrategyModifiers()
        # Проверяем, что поля существуют
        assert hasattr(modifiers, 'evolutionary_transformer_confidence_multiplier')
        assert hasattr(modifiers, 'evolutionary_transformer_strength_multiplier')
        assert hasattr(modifiers, 'evolutionary_transformer_execution_delay_ms')
        # Проверяем значения по умолчанию
        assert modifiers.evolutionary_transformer_confidence_multiplier == 1.25
        assert modifiers.evolutionary_transformer_strength_multiplier == 1.2
        assert modifiers.evolutionary_transformer_execution_delay_ms == 0
    def test_evolutionary_transformer_error_handling(self, agent_context) -> None:
        """Тест обработки ошибок в эволюционном трансформере."""
        # Устанавливаем некорректный результат
        agent_context.evolutionary_transformer_result = {"invalid_key": "invalid_value"}
        # Применяем модификатор - не должно вызывать исключение
        try:
            agent_context.apply_evolutionary_transformer_modifier()
        except Exception as e:
            pytest.fail(f"Модификатор вызвал исключение: {e}")
    @pytest.mark.asyncio
    async def test_evolutionary_transformer_cache_behavior(self, trading_orchestrator) -> None:
        """Тест поведения кэша эволюционного трансформера."""
        symbols = ["BTCUSDT"]
        # Первый вызов - должен обновить кэш
        with patch.object(trading_orchestrator, '_get_market_data_for_evolutionary_transformer') as mock_get_data:
            mock_get_data.return_value = Mock()
            with patch.object(trading_orchestrator.evolutionary_transformer, 'evolve') as mock_evolve:
                mock_evolve.return_value = Mock()
                await trading_orchestrator._update_evolutionary_transformer(symbols)
                # Проверяем, что кэш обновлен
                assert "BTCUSDT_evolutionary_transformer" in trading_orchestrator._evolutionary_transformer_cache
        # Второй вызов в течение 5 минут - не должен обновлять кэш
        with patch.object(trading_orchestrator, '_get_market_data_for_evolutionary_transformer') as mock_get_data:
            mock_get_data.return_value = Mock()
            with patch.object(trading_orchestrator.evolutionary_transformer, 'evolve') as mock_evolve:
                await trading_orchestrator._update_evolutionary_transformer(symbols)
                # Проверяем, что evolve не был вызван повторно
                assert mock_evolve.call_count == 1  # Только первый вызов 
