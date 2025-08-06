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
        return None
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
        # Создаем моки для всех зависимостей
        # Настраиваем моки
        # Arrange
        # Act
        # Assert
        # Проверяем, что EntanglementDetector был вызван
        # Проверяем, что кэш запутанности был обновлен
        # Arrange
        # Добавляем данные в кэш запутанности
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Проверяем, что EntanglementDetector был вызван
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Должно завершиться без исключения
        # Arrange
        # Act
        # Assert
        # Кэш должен обновиться
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
