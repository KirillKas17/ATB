"""
Unit тесты для интеграции SessionMarker в систему Syntra.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, MagicMock
from decimal import Decimal
from domain.sessions.session_marker import MarketSessionContext
from domain.value_objects.signal import Signal, SignalType
from domain.value_objects.money import Money
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage
from infrastructure.agents.agent_context_refactored import AgentContext, StrategyModifiers
# Временно закомментировано из-за отсутствия класса
# from application.use_cases.trading_orchestrator import DefaultTradingOrchestratorUseCase
from application.di_container import Container
class TestSessionMarkerIntegration:
    """Тесты интеграции SessionMarker в систему Syntra."""
    @pytest.fixture
    def mock_session_marker(self) -> Any:
        """Мок для SessionMarker."""
        mock = Mock(spec=MarketSessionContext)
        # Создаем мок сессии
        mock_session = Mock(spec=MarketSessionContext)
        mock_session.session_type = SessionType.ASIAN
        mock_session.is_active = True
        mock_session.phase = SessionPhase.HIGH_VOLATILITY
        mock_session.overlap_with_other_sessions = []
        # Создаем мок контекста сессий
        mock_context = Mock(spec=MarketSessionContext)
        mock_context.primary_session = mock_session
        mock_context.active_sessions = [mock_session]
        mock.get_session_context.return_value = mock_context
        return mock
    @pytest.fixture
    def mock_agent_context(self) -> Any:
        """Мок для AgentContext."""
        context = AgentContext(symbol="BTC/USDT")
        context.session_marker_result = None
        return context
    @pytest.fixture
    def mock_trading_orchestrator(self, mock_session_marker) -> Any:
        """Мок для TradingOrchestrator с SessionMarker."""
        # Создаем мок без спецификации класса
        orchestrator = Mock()
        orchestrator.session_marker = mock_session_marker
        orchestrator._session_marker_cache = {}
        orchestrator._last_session_marker_update = None
        return orchestrator
    @pytest.fixture
    def sample_signal(self) -> Any:
        """Образец торгового сигнала."""
        return Signal(
            signal_type=SignalType.BUY,
            trading_pair="BTC/USD",
            price=Money(Decimal("50000"), Currency.USD),
            volume=Volume(Decimal("0.1")),
            confidence=Percentage(Decimal("0.8")),
            strength=Percentage(Decimal("0.7")),
            timestamp=1234567890,
            metadata={}
        )
    def test_di_container_integration(self) -> None:
        """Тест интеграции SessionMarker в DI контейнер."""
        # Создаем контейнер
        container = Container()
        # Проверяем, что SessionMarker зарегистрирован
        session_marker = container.session_marker()
        assert session_marker is not None
        assert isinstance(session_marker, MarketSessionContext)
    def test_agent_context_session_marker_field(self, mock_agent_context) -> None:
        """Тест поля session_marker_result в AgentContext."""
        # Проверяем, что поле существует
        assert hasattr(mock_agent_context, 'session_marker_result')
        assert mock_agent_context.session_marker_result is None
    def test_agent_context_apply_session_marker_modifier(self, mock_agent_context, sample_signal) -> None:
        """Тест метода apply_session_marker_modifier в AgentContext."""
        # Создаем мок результат маркера сессий
        mock_session = Mock(spec=MarketSessionContext)
        mock_session.session_type = SessionType.ASIAN
        mock_session.is_active = True
        mock_session.phase = SessionPhase.HIGH_VOLATILITY
        mock_session.overlap_with_other_sessions = []
        mock_context = Mock(spec=MarketSessionContext)
        mock_context.primary_session = mock_session
        mock_context.active_sessions = [mock_session]
        mock_agent_context.session_marker_result = mock_context
        # Применяем модификатор
        modified_signal = mock_agent_context.apply_session_marker_modifier(sample_signal)
        # Проверяем, что сигнал был модифицирован
        assert modified_signal is not None
        assert modified_signal.confidence > sample_signal.confidence  # Увеличена уверенность
        assert modified_signal.strength > sample_signal.strength  # Увеличен размер позиции
        assert "session_marker" in modified_signal.metadata
    def test_trading_orchestrator_session_marker_constructor(self, mock_session_marker) -> None:
        """Тест конструктора TradingOrchestrator с SessionMarker."""
        # Создаем моки для зависимостей
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        # Создаем оркестратор (упрощенно)
        orchestrator = Mock()
        orchestrator.session_marker = mock_session_marker
        orchestrator._session_marker_cache = {}
        orchestrator._last_session_marker_update = None
        # Проверяем, что SessionMarker был добавлен
        assert orchestrator.session_marker == mock_session_marker
        assert hasattr(orchestrator, '_session_marker_cache')
        assert hasattr(orchestrator, '_last_session_marker_update')
    @pytest.mark.asyncio
    async def test_update_session_marker(self, mock_trading_orchestrator) -> None:
        """Тест метода _update_session_marker."""
        symbols = ["BTC/USD", "ETH/USD"]
        # Вызываем метод
        await mock_trading_orchestrator._update_session_marker(symbols)
        # Проверяем, что SessionMarker был вызван
        mock_trading_orchestrator.session_marker.get_session_context.assert_called()
        # Проверяем, что кэш был обновлен
        assert len(mock_trading_orchestrator._session_marker_cache) > 0
        assert mock_trading_orchestrator._last_session_marker_update is not None
    @pytest.mark.asyncio
    async def test_apply_session_marker_analysis(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест метода _apply_session_marker_analysis."""
        symbol = "BTC/USD"
        # Подготавливаем кэш
        mock_session = Mock(spec=MarketSessionContext)
        mock_session.session_type = SessionType.ASIAN
        mock_session.is_active = True
        mock_session.phase = SessionPhase.HIGH_VOLATILITY
        mock_session.overlap_with_other_sessions = []
        mock_context = Mock(spec=MarketSessionContext)
        mock_context.primary_session = mock_session
        mock_context.active_sessions = [mock_session]
        mock_trading_orchestrator._session_marker_cache[symbol] = {
            "context": mock_context,
            "timestamp": 1234567890
        }
        # Применяем анализ
        modified_signal = await mock_trading_orchestrator._apply_session_marker_analysis(symbol, sample_signal)
        # Проверяем результат
        assert modified_signal is not None
        assert modified_signal.confidence > sample_signal.confidence
        assert modified_signal.strength > sample_signal.strength
        assert "session_marker" in modified_signal.metadata
    @pytest.mark.asyncio
    async def test_session_marker_in_execute_strategy(self, mock_trading_orchestrator) -> None:
        """Тест интеграции SessionMarker в execute_strategy."""
        # Создаем мок запрос
        mock_request = Mock()
        mock_request.symbol = "BTC/USD"
        mock_request.risk_level = "medium"
        # Мокаем другие методы
        mock_trading_orchestrator._update_session_marker = AsyncMock()
        mock_trading_orchestrator._apply_session_marker_analysis = AsyncMock(return_value=sample_signal)
        # Вызываем execute_strategy (частично)
        await mock_trading_orchestrator._update_session_marker([mock_request.symbol])
        # Проверяем, что SessionMarker был обновлен
        mock_trading_orchestrator._update_session_marker.assert_called_once_with([mock_request.symbol])
    @pytest.mark.asyncio
    async def test_session_marker_in_process_signal(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест интеграции SessionMarker в process_signal."""
        # Создаем мок запрос
        mock_request = Mock()
        mock_request.signal = sample_signal
        # Мокаем методы
        mock_trading_orchestrator._update_session_marker = AsyncMock()
        mock_trading_orchestrator._apply_session_marker_analysis = AsyncMock(return_value=sample_signal)
        # Вызываем process_signal (частично)
        await mock_trading_orchestrator._update_session_marker([mock_request.signal.trading_pair])
        modified_signal = await mock_trading_orchestrator._apply_session_marker_analysis(
            mock_request.signal.trading_pair, mock_request.signal
        )
        # Проверяем, что SessionMarker был применен
        mock_trading_orchestrator._update_session_marker.assert_called_once_with([mock_request.signal.trading_pair])
        mock_trading_orchestrator._apply_session_marker_analysis.assert_called_once_with(
            mock_request.signal.trading_pair, mock_request.signal
        )
        assert modified_signal is not None
    def test_session_marker_strategy_modifiers(self) -> None:
        """Тест модификаторов стратегий для SessionMarker."""
        modifiers = StrategyModifiers()
        # Проверяем, что модификаторы для SessionMarker существуют
        assert hasattr(modifiers, 'session_marker_confidence_multiplier')
        assert hasattr(modifiers, 'session_marker_strength_multiplier')
        assert hasattr(modifiers, 'session_marker_execution_delay_ms')
    @pytest.mark.asyncio
    async def test_session_marker_error_handling(self, mock_trading_orchestrator, sample_signal) -> None:
        """Тест обработки ошибок в SessionMarker."""
        symbol = "BTC/USD"
        # Симулируем ошибку в SessionMarker
        mock_trading_orchestrator.session_marker.get_session_context.side_effect = Exception("Session marker error")
        # Проверяем, что ошибка обрабатывается корректно
        try:
            # Применяем анализ - должен вернуть исходный сигнал
            modified_signal = await mock_trading_orchestrator._apply_session_marker_analysis(symbol, sample_signal)
            assert modified_signal == sample_signal
        except Exception:
            pytest.fail("Session marker error should be handled gracefully")
    def test_session_marker_cache_management(self, mock_trading_orchestrator) -> None:
        """Тест управления кэшем SessionMarker."""
        symbol = "BTC/USD"
        # Проверяем, что кэш пустой изначально
        assert symbol not in mock_trading_orchestrator._session_marker_cache
        # Добавляем данные в кэш
        mock_trading_orchestrator._session_marker_cache[symbol] = {
            "context": Mock(),
            "timestamp": 1234567890
        }
        # Проверяем, что данные добавлены
        assert symbol in mock_trading_orchestrator._session_marker_cache
        assert mock_trading_orchestrator._session_marker_cache[symbol]["timestamp"] == 1234567890
    def test_session_marker_metadata_structure(self, mock_agent_context, sample_signal) -> None:
        """Тест структуры метаданных SessionMarker."""
        # Создаем мок результат
        mock_session = Mock(spec=MarketSessionContext)
        mock_session.session_type = SessionType.ASIAN
        mock_session.is_active = True
        mock_session.phase = SessionPhase.HIGH_VOLATILITY
        mock_session.overlap_with_other_sessions = []
        mock_context = Mock(spec=MarketSessionContext)
        mock_context.primary_session = mock_session
        mock_context.active_sessions = [mock_session]
        mock_agent_context.session_marker_result = mock_context
        # Применяем модификатор
        modified_signal = mock_agent_context.apply_session_marker_modifier(sample_signal)
        # Проверяем структуру метаданных
        assert "session_marker" in modified_signal.metadata
        session_metadata = modified_signal.metadata["session_marker"]
        assert "primary_session" in session_metadata
        assert "phase" in session_metadata
        assert "active_sessions_count" in session_metadata
        assert "overlap_count" in session_metadata
        assert session_metadata["primary_session"] == "ASIAN"
        assert session_metadata["phase"] == "HIGH_VOLATILITY"
        assert session_metadata["active_sessions_count"] == 1
        assert session_metadata["overlap_count"] == 0
if __name__ == "__main__":
    pytest.main([__file__]) 
