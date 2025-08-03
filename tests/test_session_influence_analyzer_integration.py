"""
Unit тесты для интеграции SessionInfluenceAnalyzer.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from domain.sessions.session_influence_analyzer import SessionInfluenceAnalyzer, SessionInfluenceResult
from infrastructure.agents.agent_context_refactored import AgentContext
# Временно закомментировано из-за отсутствия класса
# from application.use_cases.trading_orchestrator import TradingOrchestratorUseCase
from domain.entities.strategy import Signal, SignalType
class TestSessionInfluenceAnalyzerIntegration:
    """Тесты интеграции SessionInfluenceAnalyzer."""
    @pytest.fixture
    def session_influence_analyzer(self) -> Any:
        """Фикстура для SessionInfluenceAnalyzer."""
        return SessionInfluenceAnalyzer()
    @pytest.fixture
    def agent_context(self) -> Any:
        """Фикстура для AgentContext."""
        return AgentContext(symbol="BTCUSDT")
    @pytest.fixture
    def mock_orderbook_data(self) -> Any:
        """Фикстура для данных ордербука."""
        return {
            "bids": [
                [50000.0, 1.5],
                [49999.0, 2.0],
                [49998.0, 1.0]
            ],
            "asks": [
                [50001.0, 1.2],
                [50002.0, 1.8],
                [50003.0, 0.8]
            ],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
    @pytest.fixture
    def mock_trading_orchestrator(self) -> Any:
        """Фикстура для TradingOrchestrator."""
        return Mock(spec=TradingOrchestratorUseCase)
    def test_session_influence_analyzer_creation(self, session_influence_analyzer) -> None:
        """Тест создания SessionInfluenceAnalyzer."""
        assert session_influence_analyzer is not None
        assert hasattr(session_influence_analyzer, 'analyze_session_influence')
    def test_session_influence_analyzer_analysis(self, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест анализа влияния сессий."""
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        assert isinstance(result, SessionInfluenceResult)
        assert hasattr(result, 'session_type')
        assert hasattr(result, 'influence_score')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'volume_impact')
        assert hasattr(result, 'volatility_impact')
        assert hasattr(result, 'liquidity_impact')
        assert hasattr(result, 'price_impact')
        assert hasattr(result, 'timestamp')
    def test_agent_context_session_influence_field(self, agent_context) -> None:
        """Тест наличия поля session_influence_result в AgentContext."""
        assert hasattr(agent_context, 'session_influence_result')
        assert agent_context.session_influence_result is None
    def test_agent_context_session_influence_methods(self, agent_context) -> None:
        """Тест методов работы с session_influence_result в AgentContext."""
        # Проверяем наличие методов
        assert hasattr(agent_context, 'apply_session_influence_modifier')
        assert hasattr(agent_context, 'get_session_influence_status')
        assert hasattr(agent_context, 'update_session_influence_result')
        assert hasattr(agent_context, 'get_session_influence_result')
        # Проверяем, что методы вызываются без ошибок
        agent_context.apply_session_influence_modifier()
        status = agent_context.get_session_influence_status()
        assert status["influence_detected"] is False
        assert status["status"] == "unknown"
    def test_agent_context_session_influence_modifier(self, agent_context, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест применения модификатора влияния сессий в AgentContext."""
        # Создаем результат анализа
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        # Обновляем контекст
        agent_context.update_session_influence_result(result)
        # Сохраняем исходные значения модификаторов
        original_aggressiveness = agent_context.strategy_modifiers.order_aggressiveness
        original_position_size = agent_context.strategy_modifiers.position_size_multiplier
        original_confidence = agent_context.strategy_modifiers.confidence_multiplier
        # Применяем модификатор
        agent_context.apply_session_influence_modifier()
        # Проверяем, что модификаторы изменились
        assert agent_context.strategy_modifiers.order_aggressiveness != original_aggressiveness
        assert agent_context.strategy_modifiers.position_size_multiplier != original_position_size
        assert agent_context.strategy_modifiers.confidence_multiplier != original_confidence
    def test_agent_context_session_influence_status(self, agent_context, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест получения статуса влияния сессий в AgentContext."""
        # Создаем результат анализа
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        # Обновляем контекст
        agent_context.update_session_influence_result(result)
        # Получаем статус
        status = agent_context.get_session_influence_status()
        assert status["influence_detected"] is True
        assert "session_type" in status
        assert "influence_score" in status
        assert "confidence" in status
        assert "volume_impact" in status
        assert "volatility_impact" in status
        assert "liquidity_impact" in status
        assert "price_impact" in status
        assert "timestamp" in status
        assert "status" in status
    @pytest.mark.asyncio
    async def test_trading_orchestrator_session_influence_integration(self, mock_trading_orchestrator, session_influence_analyzer) -> None:
        """Тест интеграции SessionInfluenceAnalyzer в TradingOrchestrator."""
        # Проверяем, что SessionInfluenceAnalyzer добавлен в конструктор
        assert hasattr(mock_trading_orchestrator, 'session_influence_analyzer')
        # Проверяем наличие кэша
        assert hasattr(mock_trading_orchestrator, '_session_influence_cache')
        assert hasattr(mock_trading_orchestrator, '_last_session_influence_update')
    @pytest.mark.asyncio
    async def test_trading_orchestrator_session_influence_methods(self, mock_trading_orchestrator) -> None:
        """Тест методов работы с SessionInfluenceAnalyzer в TradingOrchestrator."""
        # Проверяем наличие методов
        assert hasattr(mock_trading_orchestrator, '_update_session_influence_analysis')
        assert hasattr(mock_trading_orchestrator, '_get_orderbook_data_for_session_influence_analysis')
        assert hasattr(mock_trading_orchestrator, '_apply_session_influence_analysis')
    def test_session_influence_result_serialization(self, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест сериализации результата анализа влияния сессий."""
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        # Проверяем метод to_dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "session_type" in result_dict
        assert "influence_score" in result_dict
        assert "confidence" in result_dict
        assert "volume_impact" in result_dict
        assert "volatility_impact" in result_dict
        assert "liquidity_impact" in result_dict
        assert "price_impact" in result_dict
        assert "timestamp" in result_dict
    def test_session_influence_analyzer_with_different_session_types(self, session_influence_analyzer) -> None:
        """Тест анализа влияния сессий для разных типов сессий."""
        # Тестируем для азиатской сессии
        asian_session_data = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50001.0, 1.0]],
            "timestamp": "2024-01-01T02:00:00",  # Азиатское время
            "symbol": "BTCUSDT"
        }
        result = session_influence_analyzer.analyze_session_influence(asian_session_data)
        assert result.session_type in ["asian", "london", "new_york", "overlap"]
    def test_session_influence_analyzer_confidence_calculation(self, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест расчета уверенности в анализе влияния сессий."""
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        # Проверяем, что уверенность в разумных пределах
        assert 0.0 <= result.confidence <= 1.0
    def test_session_influence_analyzer_impact_calculations(self, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест расчета различных типов влияния."""
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        # Проверяем, что все метрики влияния в разумных пределах
        assert 0.0 <= result.volume_impact <= 1.0
        assert 0.0 <= result.volatility_impact <= 1.0
        assert 0.0 <= result.liquidity_impact <= 1.0
        assert 0.0 <= result.price_impact <= 1.0
    def test_agent_context_session_influence_with_high_influence(self, agent_context, session_influence_analyzer) -> None:
        """Тест AgentContext с высоким влиянием сессий."""
        # Создаем данные с высоким влиянием
        high_influence_data = {
            "bids": [[50000.0, 10.0]],  # Большой объем
            "asks": [[50001.0, 10.0]],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
        result = session_influence_analyzer.analyze_session_influence(high_influence_data)
        agent_context.update_session_influence_result(result)
        # Сохраняем исходные значения
        original_aggressiveness = agent_context.strategy_modifiers.order_aggressiveness
        original_risk = agent_context.strategy_modifiers.risk_multiplier
        # Применяем модификатор
        agent_context.apply_session_influence_modifier()
        # Проверяем, что модификаторы изменились
        assert agent_context.strategy_modifiers.order_aggressiveness != original_aggressiveness
        assert agent_context.strategy_modifiers.risk_multiplier != original_risk
    def test_agent_context_session_influence_with_low_influence(self, agent_context, session_influence_analyzer) -> None:
        """Тест AgentContext с низким влиянием сессий."""
        # Создаем данные с низким влиянием
        low_influence_data = {
            "bids": [[50000.0, 0.1]],  # Малый объем
            "asks": [[50001.0, 0.1]],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
        result = session_influence_analyzer.analyze_session_influence(low_influence_data)
        agent_context.update_session_influence_result(result)
        # Сохраняем исходные значения
        original_aggressiveness = agent_context.strategy_modifiers.order_aggressiveness
        original_position_size = agent_context.strategy_modifiers.position_size_multiplier
        # Применяем модификатор
        agent_context.apply_session_influence_modifier()
        # Проверяем, что модификаторы изменились
        assert agent_context.strategy_modifiers.order_aggressiveness != original_aggressiveness
        assert agent_context.strategy_modifiers.position_size_multiplier != original_position_size
    def test_session_influence_analyzer_edge_cases(self, session_influence_analyzer) -> None:
        """Тест граничных случаев для SessionInfluenceAnalyzer."""
        # Тест с пустыми данными
        empty_data = {
            "bids": [],
            "asks": [],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
        result = session_influence_analyzer.analyze_session_influence(empty_data)
        assert isinstance(result, SessionInfluenceResult)
        # Тест с очень большими объемами
        large_volume_data = {
            "bids": [[50000.0, 1000000.0]],
            "asks": [[50001.0, 1000000.0]],
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT"
        }
        result = session_influence_analyzer.analyze_session_influence(large_volume_data)
        assert isinstance(result, SessionInfluenceResult)
    def test_session_influence_analyzer_performance(self, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест производительности SessionInfluenceAnalyzer."""
        import time
        start_time = time.time()
        # Выполняем анализ несколько раз
        for _ in range(100):
            result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
            assert isinstance(result, SessionInfluenceResult)
        end_time = time.time()
        execution_time = end_time - start_time
        # Проверяем, что анализ выполняется достаточно быстро
        assert execution_time < 1.0  # Менее 1 секунды для 100 анализов
    def test_session_influence_analyzer_integration_with_signal(self, session_influence_analyzer, mock_orderbook_data) -> None:
        """Тест интеграции SessionInfluenceAnalyzer с торговыми сигналами."""
        # Создаем торговый сигнал
        signal = Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTCUSDT",
            confidence=0.8,
            strength=0.7,
            timestamp=datetime.now()
        )
        # Анализируем влияние сессий
        result = session_influence_analyzer.analyze_session_influence(mock_orderbook_data)
        # Проверяем, что результат можно использовать для модификации сигнала
        assert result.influence_score >= 0.0
        assert result.confidence >= 0.0
        # Симулируем модификацию сигнала на основе результата
        if result.influence_score > 0.7:
            signal.confidence *= 0.8  # Снижаем уверенность при высоком влиянии
        elif result.influence_score < 0.3:
            signal.confidence *= 1.1  # Увеличиваем уверенность при низком влиянии
        assert signal.confidence > 0.0
        assert signal.confidence <= 1.0
if __name__ == "__main__":
    pytest.main([__file__]) 
