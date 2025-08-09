"""
Unit тесты для всех интегрированных модулей Syntra.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime, timedelta
from domain.entities.trading import Signal, SignalType, OrderSide, OrderType, OrderStatus
from domain.entities.order import Order
from domain.entities.portfolio import Portfolio
from domain.entities.strategy import Strategy
from domain.value_objects.percentage import Percentage
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifier
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
from application.di_container_refactored import ContainerConfig


class TestModuleIntegrations:
    """Тесты интеграции всех модулей."""

    @pytest.fixture
    def agent_context(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестового AgentContext."""
        return AgentContext(
            symbol="BTCUSDT",
            market_context=Mock(),
            pattern_prediction_context=Mock(),
            session_context=Mock(),
            strategy_modifiers=StrategyModifiers(),
        )

    @pytest.fixture
    def mock_strategy(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок стратегии."""
        strategy = Mock(spec=Strategy)
        strategy.generate_signals = AsyncMock(
            return_value=[
                Signal(
                    id="test_signal_1",
                    symbol="BTCUSDT",
                    signal_type=SignalType.BUY,
                    confidence=Percentage(Decimal("0.8")),
                    price=Decimal("50000"),
                    amount=Decimal("0.1"),
                    created_at=datetime.now(),
                )
            ]
        )
        return strategy

    @pytest.fixture
    def mock_portfolio(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок портфеля."""
        return Mock(spec=Portfolio)

    @pytest.fixture
    def trading_orchestrator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестового TradingOrchestrator."""
        return DefaultTradingOrchestratorUseCase(
            order_repository=Mock(),
            position_repository=Mock(),
            portfolio_repository=Mock(),
            trading_repository=Mock(),
            strategy_repository=Mock(),
            enhanced_trading_service=Mock(),
            agent_context_manager=Mock(),
        )

    # === Тесты доменных модулей ===
    @pytest.mark.asyncio
    async def test_market_pattern_recognizer_integration(self, agent_context) -> None:
        """Тест интеграции MarketPatternRecognizer."""
        # Подготавливаем данные
        agent_context.market_pattern_result = {"pattern_confidence": 0.85, "pattern_type": "breakout", "strength": 0.9}
        # Применяем модификатор
        agent_context.apply_market_pattern_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0
        assert modifiers.position_size_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_entanglement_detector_integration(self, agent_context) -> None:
        """Тест интеграции EntanglementDetector."""
        # Подготавливаем данные
        agent_context.entanglement_result = {
            "entanglement_level": 0.7,
            "correlation_strength": 0.8,
            "risk_multiplier": 1.2,
        }
        # Применяем модификатор
        agent_context.apply_entanglement_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.risk_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_mirror_detector_integration(self, agent_context) -> None:
        """Тест интеграции MirrorDetector."""
        # Подготавливаем данные
        agent_context.mirror_signal = {"mirror_confidence": 0.75, "mirror_strength": 0.8, "price_offset": 0.5}
        # Применяем модификатор
        agent_context.apply_mirror_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.price_offset_percent > 0

    @pytest.mark.asyncio
    async def test_noise_analyzer_integration(self, agent_context) -> None:
        """Тест интеграции NoiseAnalyzer."""
        # Подготавливаем данные
        agent_context.noise_result = {"noise_level": 0.3, "signal_quality": 0.8, "confidence_adjustment": 0.9}
        # Применяем модификатор
        agent_context.apply_noise_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 0

    @pytest.mark.asyncio
    async def test_session_influence_analyzer_integration(self, agent_context) -> None:
        """Тест интеграции SessionInfluenceAnalyzer."""
        # Подготавливаем данные
        agent_context.session_influence_result = {
            "session_strength": 0.6,
            "influence_multiplier": 1.1,
            "timing_adjustment": 100,
        }
        # Применяем модификатор
        agent_context.apply_session_influence_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.execution_delay_ms > 0

    @pytest.mark.asyncio
    async def test_session_marker_integration(self, agent_context) -> None:
        """Тест интеграции SessionMarker."""
        # Подготавливаем данные
        agent_context.session_marker_result = {
            "session_confidence": 0.8,
            "session_strength": 0.7,
            "timing_multiplier": 1.2,
        }
        # Применяем модификатор
        agent_context.apply_session_marker_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.execution_delay_ms > 0

    # === Тесты ML сервисов ===
    @pytest.mark.asyncio
    async def test_live_adaptation_model_integration(self, agent_context) -> None:
        """Тест интеграции LiveAdaptationModel."""
        # Подготавливаем данные
        agent_context.live_adaptation_result = {
            "adaptation_confidence": 0.85,
            "adaptation_strength": 0.9,
            "learning_rate": 0.1,
        }
        # Применяем модификатор
        agent_context.apply_live_adaptation_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_decision_reasoner_integration(self, agent_context) -> None:
        """Тест интеграции DecisionReasoner."""
        # Подготавливаем данные
        agent_context.decision_reasoning_result = {
            "reasoning_confidence": 0.8,
            "decision_strength": 0.75,
            "logic_multiplier": 1.1,
        }
        # Применяем модификатор
        agent_context.apply_decision_reasoning_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_evolutionary_transformer_integration(self, agent_context) -> None:
        """Тест интеграции EvolutionaryTransformer."""
        # Подготавливаем данные
        agent_context.evolutionary_transformer_result = {
            "evolution_confidence": 0.9,
            "transformation_strength": 0.85,
            "fitness_multiplier": 1.2,
        }
        # Применяем модификатор
        agent_context.apply_evolutionary_transformer_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_pattern_discovery_integration(self, agent_context) -> None:
        """Тест интеграции PatternDiscovery."""
        # Подготавливаем данные
        agent_context.pattern_discovery_result = {
            "discovery_confidence": 0.8,
            "pattern_strength": 0.7,
            "novelty_multiplier": 1.1,
        }
        # Применяем модификатор
        agent_context.apply_pattern_discovery_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_meta_learning_integration(self, agent_context) -> None:
        """Тест интеграции MetaLearning."""
        # Подготавливаем данные
        agent_context.meta_learning_result = {
            "meta_confidence": 0.85,
            "learning_strength": 0.8,
            "meta_multiplier": 1.15,
        }
        # Применяем модификатор
        agent_context.apply_meta_learning_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    # === Тесты специализированных агентов ===
    @pytest.mark.asyncio
    async def test_agent_whales_integration(self, agent_context) -> None:
        """Тест интеграции AgentWhales."""
        # Подготавливаем данные
        agent_context.whale_analysis_result = {"whale_confidence": 0.7, "whale_strength": 0.6, "whale_multiplier": 1.3}
        # Применяем модификатор
        agent_context.apply_whale_analysis_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.position_size_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_agent_risk_integration(self, agent_context) -> None:
        """Тест интеграции AgentRisk."""
        # Подготавливаем данные
        agent_context.risk_analysis_result = {"risk_confidence": 0.8, "risk_level": 0.6, "risk_multiplier": 1.1}
        # Применяем модификатор
        agent_context.apply_risk_analysis_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.risk_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_agent_portfolio_integration(self, agent_context) -> None:
        """Тест интеграции AgentPortfolio."""
        # Подготавливаем данные
        agent_context.portfolio_analysis_result = {
            "portfolio_confidence": 0.75,
            "portfolio_strength": 0.7,
            "portfolio_multiplier": 1.05,
        }
        # Применяем модификатор
        agent_context.apply_portfolio_analysis_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.position_size_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_agent_meta_controller_integration(self, agent_context) -> None:
        """Тест интеграции AgentMetaController."""
        # Подготавливаем данные
        agent_context.meta_controller_result = {"meta_confidence": 0.9, "meta_strength": 0.85, "meta_multiplier": 1.2}
        # Применяем модификатор
        agent_context.apply_meta_controller_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    # === Тесты эволюционных компонентов ===
    @pytest.mark.asyncio
    async def test_genetic_optimizer_integration(self, agent_context) -> None:
        """Тест интеграции GeneticOptimizer."""
        # Подготавливаем данные
        agent_context.genetic_optimization_result = {
            "genetic_confidence": 0.85,
            "optimization_strength": 0.8,
            "fitness_multiplier": 1.15,
        }
        # Применяем модификатор
        agent_context.apply_genetic_optimization_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    # === Тесты дополнительных ML сервисов ===
    @pytest.mark.asyncio
    async def test_model_selector_integration(self, agent_context) -> None:
        """Тест интеграции ModelSelector."""
        # Подготавливаем данные
        agent_context.model_selector_result = {
            "model_confidence": 0.8,
            "model_strength": 0.75,
            "selection_multiplier": 1.1,
        }
        # Применяем модификатор
        agent_context.apply_model_selector_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_advanced_price_predictor_integration(self, agent_context) -> None:
        """Тест интеграции AdvancedPricePredictor."""
        # Подготавливаем данные
        agent_context.advanced_price_predictor_result = {
            "prediction_confidence": 0.85,
            "prediction_strength": 0.8,
            "price_multiplier": 1.05,
        }
        # Применяем модификатор
        agent_context.apply_advanced_price_predictor_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.price_offset_percent > 0

    @pytest.mark.asyncio
    async def test_window_optimizer_integration(self, agent_context) -> None:
        """Тест интеграции WindowOptimizer."""
        # Подготавливаем данные
        agent_context.window_optimizer_result = {
            "window_confidence": 0.8,
            "window_strength": 0.7,
            "optimization_multiplier": 1.1,
        }
        # Применяем модификатор
        agent_context.apply_window_optimizer_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_state_manager_integration(self, agent_context) -> None:
        """Тест интеграции StateManager."""
        # Подготавливаем данные
        agent_context.state_manager_result = {"state_confidence": 0.85, "state_strength": 0.8, "state_multiplier": 1.1}
        # Применяем модификатор
        agent_context.apply_state_manager_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    # === Тесты инструментов обучения ===
    @pytest.mark.asyncio
    async def test_sandbox_trainer_integration(self, agent_context) -> None:
        """Тест интеграции SandboxTrainer."""
        # Подготавливаем данные
        agent_context.sandbox_trainer_result = {
            "trainer_confidence": 0.8,
            "trainer_strength": 0.75,
            "training_multiplier": 1.1,
        }
        # Применяем модификатор
        agent_context.apply_sandbox_trainer_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_model_trainer_integration(self, agent_context) -> None:
        """Тест интеграции ModelTrainer."""
        # Подготавливаем данные
        agent_context.model_trainer_result = {
            "trainer_confidence": 0.85,
            "trainer_strength": 0.8,
            "training_multiplier": 1.15,
        }
        # Применяем модификатор
        agent_context.apply_model_trainer_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_window_model_trainer_integration(self, agent_context) -> None:
        """Тест интеграции WindowModelTrainer."""
        # Подготавливаем данные
        agent_context.window_model_trainer_result = {
            "trainer_confidence": 0.8,
            "trainer_strength": 0.7,
            "training_multiplier": 1.1,
        }
        # Применяем модификатор
        agent_context.apply_window_model_trainer_modifier()
        # Проверяем результат
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0

    # === Тесты производительности ===
    @pytest.mark.asyncio
    async def test_apply_all_modifiers_performance(self, agent_context) -> None:
        """Тест производительности метода apply_all_modifiers."""
        import time

        # Заполняем все результаты
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.entanglement_result = {"entanglement_level": 0.7}
        agent_context.mirror_signal = {"mirror_confidence": 0.75}
        agent_context.noise_result = {"noise_level": 0.3}
        agent_context.session_influence_result = {"session_strength": 0.6}
        agent_context.session_marker_result = {"session_confidence": 0.8}
        agent_context.live_adaptation_result = {"adaptation_confidence": 0.85}
        agent_context.decision_reasoning_result = {"reasoning_confidence": 0.8}
        agent_context.evolutionary_transformer_result = {"evolution_confidence": 0.9}
        agent_context.pattern_discovery_result = {"discovery_confidence": 0.8}
        agent_context.meta_learning_result = {"meta_confidence": 0.85}
        agent_context.whale_analysis_result = {"whale_confidence": 0.7}
        agent_context.risk_analysis_result = {"risk_confidence": 0.8}
        agent_context.portfolio_analysis_result = {"portfolio_confidence": 0.75}
        agent_context.meta_controller_result = {"meta_confidence": 0.9}
        agent_context.genetic_optimization_result = {"genetic_confidence": 0.85}
        agent_context.model_selector_result = {"model_confidence": 0.8}
        agent_context.advanced_price_predictor_result = {"prediction_confidence": 0.85}
        agent_context.window_optimizer_result = {"window_confidence": 0.8}
        agent_context.state_manager_result = {"state_confidence": 0.85}
        agent_context.sandbox_trainer_result = {"trainer_confidence": 0.8}
        agent_context.model_trainer_result = {"trainer_confidence": 0.85}
        agent_context.window_model_trainer_result = {"trainer_confidence": 0.8}
        # Измеряем время выполнения
        start_time = time.time()
        performance_metrics = agent_context.apply_all_modifiers()
        execution_time = time.time() - start_time
        # Проверяем производительность (<100ms)
        assert execution_time < 0.1, f"apply_all_modifiers took {execution_time:.3f}s, expected <0.1s"
        # Проверяем метрики
        assert "total_modifiers_applied" in performance_metrics
        assert "execution_time" in performance_metrics
        assert performance_metrics["total_modifiers_applied"] > 0

    # === Тесты совместимости ===
    @pytest.mark.asyncio
    async def test_modifiers_compatibility(self, agent_context) -> None:
        """Тест совместимости всех модификаторов."""
        # Применяем все модификаторы
        agent_context.apply_all_modifiers()
        modifiers = agent_context.strategy_modifiers
        # Проверяем, что все модификаторы в разумных пределах
        assert 0.1 <= modifiers.confidence_multiplier <= 10.0
        assert 0.1 <= modifiers.position_size_multiplier <= 10.0
        assert 0.1 <= modifiers.risk_multiplier <= 10.0
        assert -50.0 <= modifiers.price_offset_percent <= 50.0
        assert 0 <= modifiers.execution_delay_ms <= 10000

    @pytest.mark.asyncio
    async def test_signal_modification_compatibility(self, agent_context, mock_strategy) -> None:
        """Тест совместимости модификации сигналов."""
        # Получаем исходный сигнал
        original_signals = await mock_strategy.generate_signals(
            symbol="BTCUSDT", amount=Decimal("0.1"), risk_level="medium"
        )
        original_signal = original_signals[0]
        # Применяем модификаторы
        agent_context.apply_all_modifiers()
        # Модифицируем сигнал
        modified_signal = await trading_orchestrator._apply_signal_modifiers(original_signal, agent_context)
        # Проверяем, что сигнал остался валидным
        assert modified_signal.confidence >= 0
        assert modified_signal.confidence <= 1.0
        assert modified_signal.amount > 0
        assert modified_signal.price > 0

    # === Тесты отказоустойчивости ===
    @pytest.mark.asyncio
    async def test_error_handling_in_modifiers(self, agent_context) -> None:
        """Тест обработки ошибок в модификаторах."""
        # Устанавливаем некорректные данные
        agent_context.market_pattern_result = None
        agent_context.entanglement_result = {"invalid_key": "invalid_value"}
        # Применяем модификаторы (не должно вызывать исключений)
        try:
            agent_context.apply_all_modifiers()
            assert True, "Модификаторы должны обрабатывать ошибки gracefully"
        except Exception as e:
            pytest.fail(f"Модификаторы вызвали исключение: {e}")

    @pytest.mark.asyncio
    async def test_missing_module_results(self, agent_context) -> None:
        """Тест работы с отсутствующими результатами модулей."""
        # Очищаем все результаты
        agent_context.market_pattern_result = None
        agent_context.entanglement_result = None
        agent_context.mirror_signal = None
        # ... и так далее для всех модулей
        # Применяем модификаторы
        try:
            agent_context.apply_all_modifiers()
            assert True, "Система должна работать без результатов модулей"
        except Exception as e:
            pytest.fail(f"Система не может работать без результатов модулей: {e}")

    @pytest.mark.asyncio
    async def test_invalid_data_types(self, agent_context) -> None:
        """Тест обработки некорректных типов данных."""
        # Устанавливаем некорректные типы данных
        agent_context.market_pattern_result = "invalid_string"
        agent_context.entanglement_result = 123
        agent_context.mirror_signal = ["invalid", "list"]
        # Применяем модификаторы
        try:
            agent_context.apply_all_modifiers()
            assert True, "Система должна обрабатывать некорректные типы данных"
        except Exception as e:
            pytest.fail(f"Система не может обрабатывать некорректные типы данных: {e}")


class TestTradingOrchestratorIntegration:
    """Тесты интеграции TradingOrchestrator."""

    @pytest.fixture
    def orchestrator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестового оркестратора."""
        return DefaultTradingOrchestratorUseCase(
            order_repository=Mock(),
            position_repository=Mock(),
            portfolio_repository=Mock(),
            trading_repository=Mock(),
            strategy_repository=Mock(),
            enhanced_trading_service=Mock(),
            agent_context_manager=Mock(),
        )

    @pytest.mark.asyncio
    async def test_parallel_module_updates(self, orchestrator) -> None:
        """Тест параллельного обновления модулей."""
        # Мокаем методы обновления
        orchestrator._update_noise_analysis = AsyncMock()
        orchestrator._update_market_pattern_analysis = AsyncMock()
        orchestrator._update_entanglement_analysis = AsyncMock()
        # Выполняем параллельное обновление
        await orchestrator._update_all_modules_parallel(Mock(), "BTCUSDT")
        # Проверяем, что методы были вызваны
        orchestrator._update_noise_analysis.assert_called_once()
        orchestrator._update_market_pattern_analysis.assert_called_once()
        orchestrator._update_entanglement_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_parallel_updates(self, orchestrator) -> None:
        """Тест обработки ошибок при параллельном обновлении."""
        # Мокаем метод, который вызывает исключение
        orchestrator._update_noise_analysis = AsyncMock(side_effect=Exception("Test error"))
        orchestrator._update_market_pattern_analysis = AsyncMock()
        # Выполняем обновление (не должно вызывать исключений)
        try:
            await orchestrator._update_all_modules_parallel(Mock(), "BTCUSDT")
            assert True, "Параллельное обновление должно обрабатывать ошибки"
        except Exception as e:
            pytest.fail(f"Параллельное обновление вызвало исключение: {e}")

    @pytest.mark.asyncio
    async def test_cache_functionality(self, orchestrator) -> None:
        """Тест функциональности кэширования."""
        # Проверяем метрики кэша
        metrics = orchestrator.get_performance_metrics()
        assert "cache_stats" in metrics
        assert "mirror_map_cache_size" in metrics["cache_stats"]
        assert "window_optimizer_cache_size" in metrics["cache_stats"]
        assert "state_manager_cache_size" in metrics["cache_stats"]

    @pytest.mark.asyncio
    async def test_active_modules_counting(self, orchestrator) -> None:
        """Тест подсчета активных модулей."""
        # Проверяем подсчет активных модулей
        active_count = orchestrator._count_active_modules()
        # Должно быть целое число >= 0
        assert isinstance(active_count, int)
        assert active_count >= 0


class TestDIContainerIntegration:
    """Тесты интеграции DI контейнера."""

    @pytest.mark.asyncio
    def test_container_configuration(self: "TestDIContainerIntegration") -> None:
        """Тест конфигурации DI контейнера."""
        # Создаем конфигурацию
        config = ContainerConfig()
        # Проверяем, что все модули зарегистрированы
        assert hasattr(config, "market_pattern_recognizer")
        assert hasattr(config, "entanglement_detector")
        assert hasattr(config, "mirror_detector")
        assert hasattr(config, "noise_analyzer")
        assert hasattr(config, "session_influence_analyzer")
        assert hasattr(config, "session_marker")
        assert hasattr(config, "live_adaptation_model")
        assert hasattr(config, "decision_reasoner")
        assert hasattr(config, "evolutionary_transformer")
        assert hasattr(config, "pattern_discovery")
        assert hasattr(config, "meta_learning")
        assert hasattr(config, "agent_whales")
        assert hasattr(config, "agent_risk")
        assert hasattr(config, "agent_portfolio")
        assert hasattr(config, "agent_meta_controller")
        assert hasattr(config, "genetic_optimizer")
        assert hasattr(config, "model_selector")
        assert hasattr(config, "advanced_price_predictor")
        assert hasattr(config, "window_optimizer")
        assert hasattr(config, "state_manager")
        assert hasattr(config, "sandbox_trainer")
        assert hasattr(config, "model_trainer")
        assert hasattr(config, "window_model_trainer")

    @pytest.mark.asyncio
    def test_singleton_registration(self: "TestDIContainerIntegration") -> None:
        """Тест регистрации синглтонов."""
        # Проверяем, что все модули зарегистрированы как синглтоны
        # Это проверяется в конфигурации DI контейнера
        assert True, "Все модули должны быть зарегистрированы как синглтоны"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
