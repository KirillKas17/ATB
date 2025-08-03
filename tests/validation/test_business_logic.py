"""
Тесты валидации бизнес-логики Syntra.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Any, Dict

from domain.entities.trading import Signal, SignalType, OrderSide, OrderType, OrderStatus
from domain.entities.order import Order
from domain.value_objects.percentage import Percentage
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifiers
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase

class TestBusinessLogicValidation:
    """Тесты валидации бизнес-логики."""
    
    @pytest.fixture
    def agent_context(self) -> Any:
        """Создание тестового AgentContext."""
        return AgentContext(
            symbol="BTCUSDT",
            market_context=Mock(),
            pattern_prediction=Mock(),
            session_context=Mock(),
            strategy_modifiers=StrategyModifiers()
        )
    
    @pytest.fixture
    def trading_orchestrator(self) -> Any:
        """Создание тестового TradingOrchestrator."""
        return DefaultTradingOrchestratorUseCase(
            order_repository=Mock(),
            position_repository=Mock(),
            portfolio_repository=Mock(),
            trading_repository=Mock(),
            strategy_repository=Mock(),
            enhanced_trading_service=Mock()
        )
    
    # === Тесты корректности торговых решений ===
    @pytest.mark.asyncio
    async def test_trading_decision_consistency(self, agent_context) -> None:
        """Тест консистентности торговых решений."""
        # Создаем сигналы с разными типами
        buy_signal = Signal(
            id="buy_signal",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.8"),
            price=Price(Decimal("50000"), Currency.USD)
        )
        sell_signal = Signal(
            id="sell_signal",
            symbol="BTCUSDT",
            signal_type=SignalType.SELL,
            confidence=Decimal("0.7"),
            price=Price(Decimal("51000"), Currency.USD)
        )
        
        # Применяем одинаковые модификаторы
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.entanglement_result = {"entanglement_level": 0.7}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигналы
        modified_buy = await self._apply_signal_modifiers(buy_signal, agent_context)
        modified_sell = await self._apply_signal_modifiers(sell_signal, agent_context)
        
        # Проверяем консистентность
        assert modified_buy.signal_type == SignalType.BUY
        assert modified_sell.signal_type == SignalType.SELL
        assert modified_buy.confidence > 0
        assert modified_sell.confidence > 0
    
    @pytest.mark.asyncio
    async def test_confidence_bounds_validation(self, agent_context) -> None:
        """Тест валидации границ уверенности."""
        # Создаем сигнал с высокой уверенностью
        high_confidence_signal = Signal(
            id="high_confidence",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.95"),
            price=Price(Decimal("50000"), Currency.USD)
        )
        
        # Применяем модификаторы, которые увеличивают уверенность
        agent_context.market_pattern_result = {"pattern_confidence": 0.9}
        agent_context.entanglement_result = {"entanglement_level": 0.8}
        agent_context.whale_analysis_result = {"whale_confidence": 0.9}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигнал
        modified_signal = await self._apply_signal_modifiers(high_confidence_signal, agent_context)
        
        # Проверяем, что уверенность не превышает 1.0
        assert modified_signal.confidence <= 1.0
        assert modified_signal.confidence > 0
    
    @pytest.mark.asyncio
    async def test_position_size_validation(self, agent_context) -> None:
        """Тест валидации размера позиции."""
        # Создаем сигнал с большим размером позиции
        large_position_signal = Signal(
            id="large_position",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.8")),
            price=Price(Decimal("50000")),
            amount=Decimal("10.0"),  # Большая позиция
            created_at=Timestamp.now()
        )
        
        # Применяем модификаторы, которые увеличивают размер позиции
        agent_context.whale_analysis_result = {"whale_confidence": 0.9}
        agent_context.portfolio_analysis_result = {"portfolio_confidence": 0.8}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигнал
        modified_signal = await self._apply_signal_modifiers(large_position_signal, agent_context)
        
        # Проверяем, что размер позиции остается положительным
        assert modified_signal.amount > 0
        assert modified_signal.amount > large_position_signal.amount
    
    @pytest.mark.asyncio
    async def test_price_modification_validation(self, agent_context) -> None:
        """Тест валидации модификации цены."""
        # Создаем сигналы с разными типами
        buy_signal = Signal(
            id="buy_signal",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.8")),
            price=Price(Decimal("50000")),
            amount=Decimal("0.1"),
            created_at=Timestamp.now()
        )
        sell_signal = Signal(
            id="sell_signal",
            symbol="BTCUSDT",
            signal_type=SignalType.SELL,
            confidence=Percentage(Decimal("0.8")),
            price=Price(Decimal("50000")),
            amount=Decimal("0.1"),
            created_at=Timestamp.now()
        )
        
        # Применяем модификаторы с ценовым смещением
        agent_context.mirror_signal = {"price_offset": 2.0}  # 2% смещение
        agent_context.advanced_price_predictor_result = {"prediction_confidence": 0.8}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигналы
        modified_buy = await self._apply_signal_modifiers(buy_signal, agent_context)
        modified_sell = await self._apply_signal_modifiers(sell_signal, agent_context)
        
        # Проверяем логику модификации цены
        assert modified_buy.price > buy_signal.price  # Цена покупки увеличивается
        assert modified_sell.price < sell_signal.price  # Цена продажи уменьшается
    
    # === Тесты валидации модификаторов стратегий ===
    @pytest.mark.asyncio
    async def test_strategy_modifiers_validation(self, agent_context) -> None:
        """Тест валидации модификаторов стратегий."""
        # Применяем все модификаторы
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
        
        # Применяем модификаторы
        performance_metrics = agent_context.apply_all_modifiers()
        
        # Проверяем модификаторы
        modifiers = agent_context.strategy_modifiers
        
        # Проверяем границы модификаторов
        assert 0.1 <= modifiers.confidence_multiplier <= 10.0
        assert 0.1 <= modifiers.position_size_multiplier <= 10.0
        assert 0.1 <= modifiers.risk_multiplier <= 10.0
        assert -50.0 <= modifiers.price_offset_percent <= 50.0
        assert 0 <= modifiers.execution_delay_ms <= 10000
        
        # Проверяем метрики
        assert "total_modifiers_applied" in performance_metrics
        assert performance_metrics["total_modifiers_applied"] > 0
        assert "execution_time" in performance_metrics
    
    @pytest.mark.asyncio
    async def test_modifier_priority_validation(self, agent_context) -> None:
        """Тест валидации приоритетов модификаторов."""
        # Применяем модификаторы с разными приоритетами
        agent_context.risk_analysis_result = {"risk_confidence": 0.9}  # Высокий приоритет
        agent_context.entanglement_result = {"entanglement_level": 0.8}  # Высокий приоритет
        agent_context.market_pattern_result = {"pattern_confidence": 0.7}  # Средний приоритет
        agent_context.whale_analysis_result = {"whale_confidence": 0.6}  # Низкий приоритет
        
        # Применяем модификаторы
        agent_context.apply_all_modifiers()
        
        # Проверяем, что модификаторы применены
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0
        assert modifiers.risk_multiplier > 1.0
    
    @pytest.mark.asyncio
    async def test_modifier_caching_validation(self, agent_context) -> None:
        """Тест валидации кэширования модификаторов."""
        # Применяем модификаторы первый раз
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.entanglement_result = {"entanglement_level": 0.7}
        first_metrics = agent_context.apply_all_modifiers()
        first_modifiers = agent_context.strategy_modifiers.copy()
        
        # Применяем модификаторы второй раз (должно использовать кэш)
        second_metrics = agent_context.apply_all_modifiers()
        second_modifiers = agent_context.strategy_modifiers
        
        # Проверяем, что результаты одинаковые
        assert first_modifiers.confidence_multiplier == second_modifiers.confidence_multiplier
        assert first_modifiers.position_size_multiplier == second_modifiers.position_size_multiplier
        assert first_modifiers.risk_multiplier == second_modifiers.risk_multiplier
        
        # Проверяем метрики
        assert "cache_hits" in first_metrics
        assert "cache_misses" in first_metrics
    
    # === Тесты совместимости с существующими модулями ===
    @pytest.mark.asyncio
    async def test_existing_module_compatibility(self, agent_context) -> None:
        """Тест совместимости с существующими модулями."""
        # Проверяем, что AgentContext совместим с существующими контекстами
        assert agent_context.market_context is not None
        assert agent_context.pattern_prediction_context is not None
        assert agent_context.session_context is not None
        
        # Проверяем, что StrategyModifiers совместимы
        modifiers = agent_context.strategy_modifiers
        assert hasattr(modifiers, 'confidence_multiplier')
        assert hasattr(modifiers, 'position_size_multiplier')
        assert hasattr(modifiers, 'risk_multiplier')
        assert hasattr(modifiers, 'price_offset_percent')
        assert hasattr(modifiers, 'execution_delay_ms')
    
    @pytest.mark.asyncio
    async def test_trading_orchestrator_compatibility(self, trading_orchestrator) -> None:
        """Тест совместимости TradingOrchestrator."""
        # Проверяем, что все модули интегрированы
        assert hasattr(trading_orchestrator, 'noise_analyzer')
        assert hasattr(trading_orchestrator, 'market_pattern_recognizer')
        assert hasattr(trading_orchestrator, 'entanglement_detector')
        assert hasattr(trading_orchestrator, 'mirror_detector')
        assert hasattr(trading_orchestrator, 'session_influence_analyzer')
        assert hasattr(trading_orchestrator, 'session_marker')
        assert hasattr(trading_orchestrator, 'live_adaptation_model')
        assert hasattr(trading_orchestrator, 'decision_reasoner')
        assert hasattr(trading_orchestrator, 'evolutionary_transformer')
        assert hasattr(trading_orchestrator, 'pattern_discovery')
        assert hasattr(trading_orchestrator, 'meta_learning')
        assert hasattr(trading_orchestrator, 'agent_whales')
        assert hasattr(trading_orchestrator, 'agent_risk')
        assert hasattr(trading_orchestrator, 'agent_portfolio')
        assert hasattr(trading_orchestrator, 'agent_meta_controller')
        assert hasattr(trading_orchestrator, 'genetic_optimizer')
        assert hasattr(trading_orchestrator, 'model_selector')
        assert hasattr(trading_orchestrator, 'advanced_price_predictor')
        assert hasattr(trading_orchestrator, 'window_optimizer')
        assert hasattr(trading_orchestrator, 'state_manager')
        assert hasattr(trading_orchestrator, 'sandbox_trainer')
        assert hasattr(trading_orchestrator, 'model_trainer')
        assert hasattr(trading_orchestrator, 'window_model_trainer')
    
    @pytest.mark.asyncio
    async def test_method_compatibility(self, trading_orchestrator) -> None:
        """Тест совместимости методов."""
        # Проверяем, что все методы обновления существуют
        update_methods = [
            '_update_noise_analysis',
            '_update_market_pattern_analysis',
            '_update_entanglement_analysis',
            '_update_mirror_detection',
            '_update_session_influence_analysis',
            '_update_session_marker',
            '_update_live_adaptation',
            '_update_decision_reasoning',
            '_update_evolutionary_transformer',
            '_update_pattern_discovery',
            '_update_meta_learning',
            '_update_whale_analysis',
            '_update_risk_analysis',
            '_update_portfolio_analysis',
            '_update_meta_controller',
            '_update_genetic_optimization',
            '_update_model_selector',
            '_update_advanced_price_predictor',
            '_update_window_optimizer',
            '_update_state_manager',
            '_update_sandbox_trainer',
            '_update_model_trainer',
            '_update_window_model_trainer'
        ]
        for method_name in update_methods:
            assert hasattr(trading_orchestrator, method_name), f"Method {method_name} not found"
    
    # === Тесты регрессии ===
    @pytest.mark.asyncio
    async def test_regression_basic_functionality(self, agent_context) -> None:
        """Тест регрессии базовой функциональности."""
        # Проверяем, что базовая функциональность не нарушена
        original_modifiers = agent_context.strategy_modifiers.copy()
        
        # Применяем модификаторы
        agent_context.apply_all_modifiers()
        
        # Проверяем, что модификаторы изменились
        assert agent_context.strategy_modifiers.confidence_multiplier != original_modifiers.confidence_multiplier
    
    @pytest.mark.asyncio
    async def test_regression_signal_processing(self, agent_context) -> None:
        """Тест регрессии обработки сигналов."""
        # Создаем базовый сигнал
        original_signal = Signal(
            id="regression_test",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.8")),
            price=Price(Decimal("50000")),
            amount=Decimal("0.1"),
            created_at=Timestamp.now()
        )
        
        # Применяем модификаторы
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигнал
        modified_signal = await self._apply_signal_modifiers(original_signal, agent_context)
        
        # Проверяем, что сигнал остался валидным
        assert modified_signal.id == original_signal.id
        assert modified_signal.symbol == original_signal.symbol
        assert modified_signal.signal_type == original_signal.signal_type
        assert modified_signal.confidence > original_signal.confidence
        assert modified_signal.amount > original_signal.amount
    
    @pytest.mark.asyncio
    async def test_regression_performance(self, agent_context) -> None:
        """Тест регрессии производительности."""
        import time
        
        # Заполняем все результаты модулей
        module_results = {
            'market_pattern_result': {"pattern_confidence": 0.8},
            'entanglement_result': {"entanglement_level": 0.7},
            'mirror_signal': {"mirror_confidence": 0.75},
            'noise_result': {"noise_level": 0.3},
            'session_influence_result': {"session_strength": 0.6},
            'session_marker_result': {"session_confidence": 0.8},
            'live_adaptation_result': {"adaptation_confidence": 0.85},
            'decision_reasoning_result': {"reasoning_confidence": 0.8},
            'evolutionary_transformer_result': {"evolution_confidence": 0.9},
            'pattern_discovery_result': {"discovery_confidence": 0.8},
            'meta_learning_result': {"meta_confidence": 0.85},
            'whale_analysis_result': {"whale_confidence": 0.7},
            'risk_analysis_result': {"risk_confidence": 0.8},
            'portfolio_analysis_result': {"portfolio_confidence": 0.75},
            'meta_controller_result': {"meta_confidence": 0.9},
            'genetic_optimization_result': {"genetic_confidence": 0.85},
            'model_selector_result': {"model_confidence": 0.8},
            'advanced_price_predictor_result': {"prediction_confidence": 0.85},
            'window_optimizer_result': {"window_confidence": 0.8},
            'state_manager_result': {"state_confidence": 0.85},
            'sandbox_trainer_result': {"trainer_confidence": 0.8},
            'model_trainer_result': {"trainer_confidence": 0.85},
            'window_model_trainer_result': {"trainer_confidence": 0.8}
        }
        
        for attr, value in module_results.items():
            setattr(agent_context, attr, value)
        
        # Измеряем время выполнения
        start_time = time.time()
        performance_metrics = agent_context.apply_all_modifiers()
        execution_time = time.time() - start_time
        
        # Проверяем производительность (<100ms)
        assert execution_time < 0.1, f"apply_all_modifiers took {execution_time:.3f}s, expected <0.1s"
        
        # Проверяем метрики
        assert "total_modifiers_applied" in performance_metrics
        assert performance_metrics["total_modifiers_applied"] > 0
    
    async def _apply_signal_modifiers(self, signal: Signal, agent_context: AgentContext) -> Signal:
        """Применение модификаторов к сигналу."""
        modifiers = agent_context.strategy_modifiers
        
        # Применяем модификаторы к сигналу
        signal.confidence *= modifiers.confidence_multiplier
        signal.confidence = min(signal.confidence, 1.0)
        
        # Модифицируем размер позиции
        if hasattr(signal, 'amount') and signal.amount:
            signal.amount *= Decimal(str(modifiers.position_size_multiplier))
        
        # Модифицируем цену
        if hasattr(signal, 'price') and signal.price:
            price_offset = modifiers.price_offset_percent / 100.0
            if signal.signal_type == SignalType.BUY:
                signal.price *= Decimal(str(1.0 + price_offset))
            else:
                signal.price *= Decimal(str(1.0 - price_offset))
        
        return signal

class TestBusinessRulesValidation:
    """Тесты валидации бизнес-правил."""
    
    @pytest.mark.asyncio
    async def test_risk_management_rules(self, agent_context) -> None:
        """Тест правил управления рисками."""
        # Создаем сигнал с высоким риском
        high_risk_signal = Signal(
            id="high_risk",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.9")),
            price=Price(Decimal("50000")),
            amount=Decimal("1.0"),  # Большая позиция
            created_at=Timestamp.now()
        )
        
        # Применяем модификаторы риска
        agent_context.risk_analysis_result = {"risk_confidence": 0.9, "risk_level": 0.8}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигнал
        modified_signal = await self._apply_signal_modifiers(high_risk_signal, agent_context)
        
        # Проверяем правила управления рисками
        assert modified_signal.confidence <= 1.0
        assert modified_signal.amount > 0
    
    @pytest.mark.asyncio
    async def test_position_sizing_rules(self, agent_context) -> None:
        """Тест правил определения размера позиции."""
        # Создаем сигналы с разными уровнями уверенности
        low_confidence_signal = Signal(
            id="low_confidence",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.3")),
            price=Price(Decimal("50000")),
            amount=Decimal("0.1"),
            created_at=Timestamp.now()
        )
        high_confidence_signal = Signal(
            id="high_confidence",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.9")),
            price=Price(Decimal("50000")),
            amount=Decimal("0.1"),
            created_at=Timestamp.now()
        )
        
        # Применяем одинаковые модификаторы
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.apply_all_modifiers()
        
        # Модифицируем сигналы
        modified_low = await self._apply_signal_modifiers(low_confidence_signal, agent_context)
        modified_high = await self._apply_signal_modifiers(high_confidence_signal, agent_context)
        
        # Проверяем правила определения размера позиции
        # Сигнал с высокой уверенностью должен иметь больший размер позиции
        assert modified_high.amount >= modified_low.amount
    
    async def _apply_signal_modifiers(self, signal: Signal, agent_context: AgentContext) -> Signal:
        """Применение модификаторов к сигналу."""
        modifiers = agent_context.strategy_modifiers
        
        # Применяем модификаторы к сигналу
        signal.confidence *= modifiers.confidence_multiplier
        signal.confidence = min(signal.confidence, 1.0)
        
        # Модифицируем размер позиции
        if hasattr(signal, 'amount') and signal.amount:
            signal.amount *= Decimal(str(modifiers.position_size_multiplier))
        
        return signal

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
