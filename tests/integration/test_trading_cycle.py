"""
Integration тесты для полного торгового цикла Syntra.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime, timedelta

from domain.entities.trading import Signal, SignalType, OrderSide, OrderType, OrderStatus
from domain.entities.order import Order
from domain.entities.portfolio import Portfolio
from domain.entities.strategy import Strategy
from domain.value_objects.percentage import Percentage
from application.use_cases.manage_orders import OrderManagementUseCase
from application.types import ExecuteStrategyRequest
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifier


class TestTradingCycleIntegration:
    """Тесты полного торгового цикла."""

    @pytest.fixture
    def mock_repositories(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок репозиториев."""
        return {
            "order_repository": Mock(),
            "position_repository": Mock(),
            "portfolio_repository": Mock(),
            "trading_repository": Mock(),
            "strategy_repository": Mock(),
        }

    @pytest.fixture
    def mock_services(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок сервисов."""
        return {
            "enhanced_trading_service": Mock(),
            "agent_context_manager": Mock(),
        }

    @pytest.fixture
    def mock_modules(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок модулей."""
        return {
            "noise_analyzer": Mock(),
            "market_pattern_recognizer": Mock(),
            "entanglement_detector": Mock(),
            "mirror_detector": Mock(),
            "session_influence_analyzer": Mock(),
            "session_marker": Mock(),
            "live_adaptation_model": Mock(),
            "decision_reasoner": Mock(),
            "evolutionary_transformer": Mock(),
            "pattern_discovery": Mock(),
            "meta_learning": Mock(),
            "agent_whales": Mock(),
            "agent_risk": Mock(),
            "agent_portfolio": Mock(),
            "agent_meta_controller": Mock(),
            "genetic_optimizer": Mock(),
            "model_selector": Mock(),
            "advanced_price_predictor": Mock(),
            "window_optimizer": Mock(),
            "state_manager": Mock(),
            "sandbox_trainer": Mock(),
            "model_trainer": Mock(),
            "window_model_trainer": Mock(),
        }

    @pytest.fixture
    def trading_orchestrator(self, mock_repositories, mock_services, mock_modules) -> Any:
        """Создание тестового TradingOrchestrator."""
        orchestrator = Mock()
        orchestrator.execute_strategy = AsyncMock()
        orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))
        orchestrator.enhanced_trading_service = mock_services["enhanced_trading_service"]
        return orchestrator

    @pytest.fixture
    def mock_strategy(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок стратегии."""
        strategy = Mock(spec=Strategy)
        strategy.id = "test_strategy_1"
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
                ),
                Signal(
                    id="test_signal_2",
                    symbol="BTCUSDT",
                    signal_type=SignalType.SELL,
                    confidence=Percentage(Decimal("0.7")),
                    price=Decimal("51000"),
                    amount=Decimal("0.05"),
                    created_at=datetime.now(),
                ),
            ]
        )
        return strategy

    @pytest.fixture
    def mock_portfolio(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание мок портфеля."""
        portfolio = Mock(spec=Portfolio)
        portfolio.id = "test_portfolio_1"
        portfolio.balance = Decimal("10000")
        portfolio.positions = {}
        return portfolio

    @pytest.fixture
    def execute_request(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание запроса на выполнение стратегии."""
        return ExecuteStrategyRequest(
            strategy_id="test_strategy_1",
            portfolio_id="test_portfolio_1",
            symbol="BTCUSDT",
            amount=Decimal("0.1"),
            risk_level="medium",
            use_sentiment_analysis=True,
        )

    @pytest.mark.asyncio
    async def test_full_trading_cycle(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест полного торгового цикла."""
        # Настраиваем моки
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=mock_strategy)
        mock_repositories["portfolio_repository"].get_by_id = AsyncMock(return_value=mock_portfolio)
        trading_orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))
        trading_orchestrator.enhanced_trading_service.get_market_sentiment_analysis = AsyncMock(
            return_value={"sentiment": "positive", "confidence": 0.8}
        )

        # Мокаем методы обновления модулей
        for module_name in trading_orchestrator.__dict__.keys():
            if module_name.startswith("_update_") and hasattr(trading_orchestrator, module_name):
                setattr(trading_orchestrator, module_name, AsyncMock())

        # Мокаем создание ордеров
        mock_order = Order(
            id="test_order_1",
            portfolio_id=mock_portfolio.id,
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Decimal("0.1"),
            price=None,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_repositories["order_repository"].create = AsyncMock(return_value=mock_order)

        # Выполняем торговый цикл
        response = await trading_orchestrator.execute_strategy(execute_request)

        # Проверяем результат
        assert response.executed is True
        assert len(response.orders_created) > 0
        assert len(response.signals_generated) > 0
        assert response.sentiment_analysis is not None
        assert "Strategy executed successfully" in response.message

    @pytest.mark.asyncio
    async def test_trading_cycle_with_real_data(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест торгового цикла с реальными данными."""
        # Настраиваем реальные данные
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=mock_strategy)
        mock_repositories["portfolio_repository"].get_by_id = AsyncMock(return_value=mock_portfolio)
        trading_orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))

        # Реальные данные сентимента
        real_sentiment = {
            "sentiment": "bullish",
            "confidence": 0.85,
            "news_score": 0.7,
            "social_score": 0.8,
            "technical_score": 0.9,
        }
        trading_orchestrator.enhanced_trading_service.get_market_sentiment_analysis = AsyncMock(
            return_value=real_sentiment
        )

        # Мокаем методы обновления модулей с реальными данными
        for module_name in trading_orchestrator.__dict__.keys():
            if module_name.startswith("_update_") and hasattr(trading_orchestrator, module_name):
                setattr(trading_orchestrator, module_name, AsyncMock())

        # Создаем реальные ордера
        def create_real_order(signal) -> Any:
            return Order(
                id=f"order_{datetime.now().timestamp()}_{signal.id}",
                portfolio_id=mock_portfolio.id,
                symbol=signal.symbol,
                order_type=(
                    OrderType.MARKET if signal.signal_type in [SignalType.BUY, SignalType.SELL] else OrderType.LIMIT
                ),
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                amount=signal.amount,
                price=signal.price,
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        mock_repositories["order_repository"].create = AsyncMock(side_effect=create_real_order)

        # Выполняем торговый цикл
        response = await trading_orchestrator.execute_strategy(execute_request)

        # Проверяем результат с реальными данными
        assert response.executed is True
        assert len(response.orders_created) == 2  # Два сигнала = два ордера
        assert len(response.signals_generated) == 2
        assert response.sentiment_analysis == real_sentiment

        # Проверяем детали ордеров
        for order in response.orders_created:
            assert order.portfolio_id == mock_portfolio.id
            assert order.symbol == "BTCUSDT"
            assert order.status == OrderStatus.PENDING
            assert order.amount > 0

    @pytest.mark.asyncio
    async def test_trading_cycle_load_test(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест нагрузки торгового цикла."""
        # Настраиваем моки
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=mock_strategy)
        mock_repositories["portfolio_repository"].get_by_id = AsyncMock(return_value=mock_portfolio)
        trading_orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))
        trading_orchestrator.enhanced_trading_service.get_market_sentiment_analysis = AsyncMock(
            return_value={"sentiment": "neutral", "confidence": 0.5}
        )

        # Мокаем методы обновления модулей
        for module_name in trading_orchestrator.__dict__.keys():
            if module_name.startswith("_update_") and hasattr(trading_orchestrator, module_name):
                setattr(trading_orchestrator, module_name, AsyncMock())

        mock_repositories["order_repository"].create = AsyncMock()

        # Выполняем множественные запросы для теста нагрузки
        import time

        start_time = time.time()

        tasks = []
        for i in range(10):  # 10 параллельных запросов
            task = asyncio.create_task(trading_orchestrator.execute_strategy(execute_request))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Проверяем результаты
        assert len(responses) == 10
        for response in responses:
            assert response.executed is True
            assert len(response.orders_created) > 0

        # Проверяем производительность (<5 секунд для 10 запросов)
        assert total_time < 5.0, f"Load test took {total_time:.2f}s, expected <5.0s"

    @pytest.mark.asyncio
    async def test_trading_cycle_error_handling(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест обработки ошибок в торговом цикле."""
        # Настраиваем моки с ошибками
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=None)  # Стратегия не найдена

        # Выполняем торговый цикл (должен вызвать исключение)
        with pytest.raises(Exception) as exc_info:
            await trading_orchestrator.execute_strategy(execute_request)

        assert "Strategy" in str(exc_info.value) and "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_trading_cycle_validation_failure(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест неудачной валидации торговых условий."""
        # Настраиваем моки
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=mock_strategy)
        mock_repositories["portfolio_repository"].get_by_id = AsyncMock(return_value=mock_portfolio)

        # Валидация не проходит
        trading_orchestrator.validate_trading_conditions = AsyncMock(
            return_value=(False, ["Insufficient balance", "Market closed"])
        )

        # Выполняем торговый цикл (должен вызвать исключение)
        with pytest.raises(Exception) as exc_info:
            await trading_orchestrator.execute_strategy(execute_request)

        assert "Trading conditions not met" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_trading_cycle_module_failures(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест отказоустойчивости при сбоях модулей."""
        # Настраиваем моки
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=mock_strategy)
        mock_repositories["portfolio_repository"].get_by_id = AsyncMock(return_value=mock_portfolio)
        trading_orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))
        trading_orchestrator.enhanced_trading_service.get_market_sentiment_analysis = AsyncMock(
            return_value={"sentiment": "positive", "confidence": 0.8}
        )

        # Некоторые модули вызывают исключения
        trading_orchestrator._update_noise_analysis = AsyncMock(side_effect=Exception("Noise analysis failed"))
        trading_orchestrator._update_market_pattern_analysis = AsyncMock(
            side_effect=Exception("Pattern analysis failed")
        )

        # Остальные модули работают нормально
        for module_name in trading_orchestrator.__dict__.keys():
            if module_name.startswith("_update_") and hasattr(trading_orchestrator, module_name):
                if not hasattr(trading_orchestrator, module_name) or not callable(
                    getattr(trading_orchestrator, module_name)
                ):
                    continue
                if module_name not in ["_update_noise_analysis", "_update_market_pattern_analysis"]:
                    setattr(trading_orchestrator, module_name, AsyncMock())

        mock_repositories["order_repository"].create = AsyncMock()

        # Выполняем торговый цикл (должен завершиться успешно несмотря на сбои модулей)
        response = await trading_orchestrator.execute_strategy(execute_request)

        # Проверяем результат
        assert response.executed is True
        assert len(response.orders_created) > 0

    @pytest.mark.asyncio
    async def test_trading_cycle_performance_metrics(
        self, trading_orchestrator, mock_strategy, mock_portfolio, execute_request, mock_repositories
    ) -> None:
        """Тест метрик производительности торгового цикла."""
        # Настраиваем моки
        mock_repositories["strategy_repository"].get_by_id = AsyncMock(return_value=mock_strategy)
        mock_repositories["portfolio_repository"].get_by_id = AsyncMock(return_value=mock_portfolio)
        trading_orchestrator.validate_trading_conditions = AsyncMock(return_value=(True, []))
        trading_orchestrator.enhanced_trading_service.get_market_sentiment_analysis = AsyncMock(
            return_value={"sentiment": "positive", "confidence": 0.8}
        )

        # Мокаем методы обновления модулей
        for module_name in trading_orchestrator.__dict__.keys():
            if module_name.startswith("_update_") and hasattr(trading_orchestrator, module_name):
                setattr(trading_orchestrator, module_name, AsyncMock())

        mock_repositories["order_repository"].create = AsyncMock()

        # Выполняем торговый цикл
        import time

        start_time = time.time()
        response = await trading_orchestrator.execute_strategy(execute_request)
        execution_time = time.time() - start_time

        # Проверяем производительность
        assert execution_time < 1.0, f"Trading cycle took {execution_time:.3f}s, expected <1.0s"
        assert response.executed is True

        # Проверяем метрики оркестратора
        metrics = trading_orchestrator.get_performance_metrics()
        assert "cache_stats" in metrics
        assert "active_modules" in metrics
        assert metrics["active_modules"] > 0


class TestSignalProcessingIntegration:
    """Тесты интеграции обработки сигналов."""

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

    @pytest.mark.asyncio
    async def test_signal_generation_with_modifiers(self, agent_context) -> None:
        """Тест генерации сигналов с модификаторами."""
        # Заполняем результаты модулей
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.entanglement_result = {"entanglement_level": 0.7}
        agent_context.noise_result = {"noise_level": 0.3}
        agent_context.whale_analysis_result = {"whale_confidence": 0.7}
        agent_context.risk_analysis_result = {"risk_confidence": 0.8}

        # Применяем модификаторы
        performance_metrics = agent_context.apply_all_modifiers()

        # Проверяем метрики
        assert "total_modifiers_applied" in performance_metrics
        assert performance_metrics["total_modifiers_applied"] > 0

        # Проверяем модификаторы
        modifiers = agent_context.strategy_modifiers
        assert modifiers.confidence_multiplier > 1.0
        assert modifiers.position_size_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_signal_modification_chain(self, agent_context) -> None:
        """Тест цепочки модификации сигналов."""
        # Создаем исходный сигнал
        original_signal = Signal(
            id="test_signal",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Percentage(Decimal("0.8")),
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            created_at=datetime.now(),
        )

        # Заполняем результаты модулей
        agent_context.market_pattern_result = {"pattern_confidence": 0.8}
        agent_context.entanglement_result = {"entanglement_level": 0.7}
        agent_context.noise_result = {"noise_level": 0.3}

        # Применяем модификаторы
        agent_context.apply_all_modifiers()

        # Модифицируем сигнал
        modified_signal = await self._apply_signal_modifiers(original_signal, agent_context)

        # Проверяем результат
        assert modified_signal.confidence > original_signal.confidence
        assert modified_signal.amount > original_signal.amount
        assert modified_signal.price > original_signal.price

    async def _apply_signal_modifiers(self, signal: Signal, agent_context) -> Signal:
        """Применение модификаторов к сигналу."""
        modifiers = agent_context.strategy_modifiers

        # Применяем модификаторы к сигналу
        signal.confidence *= modifiers.confidence_multiplier
        signal.confidence = min(signal.confidence, 1.0)

        # Модифицируем размер позиции
        if hasattr(signal, "amount") and signal.amount:
            signal.amount *= Decimal(str(modifiers.position_size_multiplier))

        # Модифицируем цену
        if hasattr(signal, "price") and signal.price:
            price_offset = modifiers.price_offset_percent / 100.0
            if signal.signal_type == SignalType.BUY:
                signal.price *= Decimal(str(1.0 + price_offset))
            else:
                signal.price *= Decimal(str(1.0 - price_offset))

        return signal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
