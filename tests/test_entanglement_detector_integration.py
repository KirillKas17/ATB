"""
Unit тесты для интеграции EntanglementDetector в TradingOrchestrator.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime
from domain.entities.strategy import Signal, SignalType, Strategy
from domain.entities.portfolio import Portfolio
from domain.intelligence.entanglement_detector import EntanglementDetector, EntanglementResult
from application.use_cases.trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    ExecuteStrategyRequest,
    ProcessSignalRequest
)
from application.di_container import DIContainer, ContainerConfig
class TestEntanglementDetectorIntegration:
    """Тесты интеграции EntanglementDetector."""
    @pytest.fixture
    def mock_entanglement_detector(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок EntanglementDetector."""
        detector = Mock(spec=EntanglementDetector)
        detector.analyze_entanglement = Mock(return_value=EntanglementResult(
            is_entangled=True,
            confidence=0.85,
            correlation_matrix={},
            lag_analysis={},
            metadata={"detection_method": "correlation_analysis"}
        ))
        return detector
    @pytest.fixture
    def mock_enhanced_trading_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок EnhancedTradingService."""
        service = Mock()
        service.get_orderbook = AsyncMock(return_value={
            "bids": [[50000, 1.0], [49999, 2.0]],
            "asks": [[50001, 1.5], [50002, 2.5]],
            "timestamp": datetime.now().isoformat()
        })
        return service
    @pytest.fixture
    def trading_orchestrator(self, mock_entanglement_detector, mock_enhanced_trading_service) -> Any:
        """TradingOrchestrator с интегрированным EntanglementDetector."""
        # Создаем моки для всех зависимостей
        order_repository = Mock()
        position_repository = Mock()
        portfolio_repository = Mock()
        trading_repository = Mock()
        strategy_repository = Mock()
        # Настраиваем моки
        portfolio = Portfolio(
            id=PortfolioId(UUID("550e8400-e29b-41d4-a716-446655440000")),
            name="Test Portfolio",
            balance=Decimal("10000"),
            currency="USD"
        )
        portfolio_repository.get_by_id = AsyncMock(return_value=portfolio)
        strategy = Strategy(
            id=StrategyId(UUID("550e8400-e29b-41d4-a716-446655440001")),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )
        strategy_repository.get_by_id = AsyncMock(return_value=strategy)
        return DefaultTradingOrchestratorUseCase(
            order_repository=order_repository,
            position_repository=position_repository,
            portfolio_repository=portfolio_repository,
            trading_repository=trading_repository,
            strategy_repository=strategy_repository,
            enhanced_trading_service=mock_enhanced_trading_service,
            entanglement_detector=mock_entanglement_detector
        )
    @pytest.mark.asyncio
    async def test_entanglement_detector_integration_in_execute_strategy(self, trading_orchestrator, mock_entanglement_detector) -> None:
        """Тест интеграции EntanglementDetector в execute_strategy."""
        # Arrange
        request = ExecuteStrategyRequest(
            strategy_id=StrategyId(UUID("550e8400-e29b-41d4-a716-446655440001")),
            portfolio_id=PortfolioId(UUID("550e8400-e29b-41d4-a716-446655440000")),
            symbol=Symbol("BTCUSDT"),
            amount=VolumeValue(Decimal("0.1"))
        )
        # Act
        with patch.object(trading_orchestrator, '_generate_signals_with_sentiment', return_value=[]):
            with patch.object(trading_orchestrator, '_create_enhanced_order_from_signal', return_value=None):
                response = await trading_orchestrator.execute_strategy(request)
        # Assert
        assert response.executed is False
        assert len(response.orders_created) == 0
        # Проверяем, что EntanglementDetector был вызван
        mock_entanglement_detector.analyze_entanglement.assert_called()
        # Проверяем, что кэш запутанности был обновлен
        assert "BTCUSDT" in trading_orchestrator._entanglement_cache
    @pytest.mark.asyncio
    async def test_entanglement_analysis_application_to_signals(self, trading_orchestrator, mock_entanglement_detector) -> None:
        """Тест применения анализа запутанности к сигналам."""
        # Arrange
        signal = Signal(
            trading_pair="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.8"),
            strength=SignalStrength.STRONG
        )
        # Добавляем данные в кэш запутанности
        trading_orchestrator._entanglement_cache["BTCUSDT"] = {
            "result": EntanglementResult(
                is_entangled=True,
                confidence=0.85,
                metadata={}
            ),
            "timestamp": datetime.now().timestamp()
        }
        # Act
        modified_signal = await trading_orchestrator._apply_entanglement_analysis("BTCUSDT", signal)
        # Assert
        assert modified_signal.confidence < signal.confidence  # Уверенность должна снизиться
        assert "execution_delay_ms" in modified_signal.metadata
    @pytest.mark.asyncio
    async def test_entanglement_detector_in_process_signal(self, trading_orchestrator, mock_entanglement_detector) -> None:
        """Тест интеграции EntanglementDetector в process_signal."""
        # Arrange
        signal = Signal(
            trading_pair="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.8"),
            strength=SignalStrength.STRONG
        )
        request = ProcessSignalRequest(
            signal=signal,
            portfolio_id=PortfolioId(UUID("550e8400-e29b-41d4-a716-446655440000")),
            auto_execute=True
        )
        # Act
        with patch.object(trading_orchestrator, '_create_enhanced_order_from_signal', return_value=None):
            response = await trading_orchestrator.process_signal(request)
        # Assert
        assert response.processed is True
        assert len(response.orders_created) == 0
        # Проверяем, что EntanglementDetector был вызван
        mock_entanglement_detector.analyze_entanglement.assert_called()
    @pytest.mark.asyncio
    async def test_entanglement_cache_update(self, trading_orchestrator, mock_entanglement_detector) -> None:
        """Тест обновления кэша запутанности."""
        # Arrange
        symbols = ["BTCUSDT", "ETHUSDT"]
        # Act
        await trading_orchestrator._update_entanglement_analysis(symbols)
        # Assert
        assert "BTCUSDT" in trading_orchestrator._entanglement_cache
        assert "ETHUSDT" in trading_orchestrator._entanglement_cache
        assert trading_orchestrator._last_entanglement_update is not None
    @pytest.mark.asyncio
    def test_entanglement_detector_disabled(self: "TestEntanglementDetectorIntegration") -> None:
        """Тест работы без EntanglementDetector."""
        # Arrange
        order_repository = Mock()
        position_repository = Mock()
        portfolio_repository = Mock()
        trading_repository = Mock()
        strategy_repository = Mock()
        enhanced_trading_service = Mock()
        orchestrator = DefaultTradingOrchestratorUseCase(
            order_repository=order_repository,
            position_repository=position_repository,
            portfolio_repository=portfolio_repository,
            trading_repository=trading_repository,
            strategy_repository=strategy_repository,
            enhanced_trading_service=enhanced_trading_service,
            entanglement_detector=None  # Отключаем детектор
        )
        signal = Signal(
            trading_pair="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.8"),
            strength=SignalStrength.STRONG
        )
        # Act
        modified_signal = await orchestrator._apply_entanglement_analysis("BTCUSDT", signal)
        # Assert
        assert modified_signal == signal  # Сигнал не должен измениться
    @pytest.mark.asyncio
    def test_entanglement_detector_in_di_container(self: "TestEntanglementDetectorIntegration") -> None:
        """Тест регистрации EntanglementDetector в DI контейнере."""
        # Arrange
        config = ContainerConfig(entanglement_detection_enabled=True)
        container = DIContainer(config)
        # Act
        detector = container.get("entanglement_detector")
        # Assert
        assert detector is not None
        assert isinstance(detector, EntanglementDetector)
    @pytest.mark.asyncio
    def test_entanglement_detector_disabled_in_di_container(self: "TestEntanglementDetectorIntegration") -> None:
        """Тест отключения EntanglementDetector в DI контейнере."""
        # Arrange
        config = ContainerConfig(entanglement_detection_enabled=False)
        container = DIContainer(config)
        # Act
        detector = container.get("entanglement_detector")
        # Assert
        assert detector is None
    @pytest.mark.asyncio
    async def test_entanglement_analysis_with_no_orderbook_data(self, trading_orchestrator, mock_enhanced_trading_service) -> None:
        """Тест анализа запутанности без данных ордербука."""
        # Arrange
        mock_enhanced_trading_service.get_orderbook = AsyncMock(return_value=None)
        # Act
        await trading_orchestrator._update_entanglement_analysis(["BTCUSDT"])
        # Assert
        assert "BTCUSDT" not in trading_orchestrator._entanglement_cache
    @pytest.mark.asyncio
    async def test_entanglement_analysis_error_handling(self, trading_orchestrator, mock_entanglement_detector) -> None:
        """Тест обработки ошибок в анализе запутанности."""
        # Arrange
        mock_entanglement_detector.analyze_entanglement.side_effect = Exception("Test error")
        # Act
        await trading_orchestrator._update_entanglement_analysis(["BTCUSDT"])
        # Assert
        # Должно завершиться без исключения
        assert "BTCUSDT" not in trading_orchestrator._entanglement_cache
    @pytest.mark.asyncio
    async def test_entanglement_cache_expiration(self, trading_orchestrator, mock_entanglement_detector) -> None:
        """Тест истечения срока действия кэша запутанности."""
        # Arrange
        trading_orchestrator._entanglement_cache["BTCUSDT"] = {
            "result": EntanglementResult(
                is_entangled=True,
                confidence=0.85,
                metadata={}
            ),
            "timestamp": 0  # Старый timestamp
        }
        # Act
        await trading_orchestrator._update_entanglement_analysis(["BTCUSDT"])
        # Assert
        # Кэш должен обновиться
        assert trading_orchestrator._entanglement_cache["BTCUSDT"]["timestamp"] > 0
class TestEntanglementDetectorPerformance:
    """Тесты производительности EntanglementDetector."""
    @pytest.mark.asyncio
    def test_entanglement_analysis_performance(self: "TestEntanglementDetectorPerformance") -> None:
        """Тест производительности анализа запутанности."""
        import time
        # Arrange
        detector = EntanglementDetector()
        orderbook_data = {
            "bids": [[50000 + i, 1.0] for i in range(100)],
            "asks": [[50001 + i, 1.0] for i in range(100)],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
        # Act
        start_time = time.time()
        result = detector.analyze_entanglement(orderbook_data)
        end_time = time.time()
        # Assert
        assert end_time - start_time < 1.0  # Анализ должен выполняться менее чем за 1 секунду
        assert isinstance(result, EntanglementResult)
        assert hasattr(result, 'is_entangled')
        assert hasattr(result, 'confidence')
class TestEntanglementDetectorIntegrationWithRealData:
    """Тесты интеграции с реальными данными."""
    @pytest.mark.asyncio
    def test_entanglement_detection_with_realistic_orderbook(self: "TestEntanglementDetectorIntegrationWithRealData") -> None:
        """Тест детекции запутанности с реалистичными данными ордербука."""
        # Arrange
        detector = EntanglementDetector()
        # Создаем реалистичные данные ордербука
        orderbook_data = {
            "bids": [
                [50000, 2.5], [49999, 1.8], [49998, 3.2], [49997, 2.1], [49996, 1.5],
                [49995, 4.0], [49994, 2.8], [49993, 1.9], [49992, 3.5], [49991, 2.3]
            ],
            "asks": [
                [50001, 2.0], [50002, 3.1], [50003, 1.7], [50004, 2.9], [50005, 2.4],
                [50006, 1.6], [50007, 3.3], [50008, 2.7], [50009, 1.8], [50010, 3.0]
            ],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
        # Act
        result = detector.analyze_entanglement(orderbook_data)
        # Assert
        assert isinstance(result, EntanglementResult)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.is_entangled, bool)
        assert isinstance(result.correlation_matrix, dict)
        assert isinstance(result.lag_analysis, dict)
        assert isinstance(result.metadata, dict) 
