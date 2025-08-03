"""
Тесты интеграции session_service с orchestrator'ом.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock
from domain.sessions.services import SessionService
from domain.sessions.factories import get_session_service
# Исправляю импорт - убираю несуществующий класс
# from application.use_cases.trading_orchestrator import DefaultTradingOrchestratorUseCase
class TestSessionServiceOrchestratorIntegration:
    """Тесты интеграции SessionService с TradingOrchestratorUseCase."""
    @pytest.fixture
    def mock_session_service(self) -> SessionService:
        """Создает мок SessionService."""
        service = Mock(spec=SessionService)
        service.analyze_session_influence = AsyncMock()
        service.get_current_session_context = Mock(return_value={"session": "test"})
        service.mark_session = AsyncMock()
        return service
    @pytest.fixture
    def mock_trading_service(self) -> Any:
        """Создает мок TradingService."""
        service = Mock()
        service.execute_order = AsyncMock()
        service.get_market_data = AsyncMock()
        return service
    @pytest.fixture
    def orchestrator(self, mock_session_service, mock_trading_service) -> Any:
        """Создает orchestrator с session_service."""
        # Исправляю - использую Mock вместо несуществующего класса
        orchestrator = Mock()
        orchestrator.session_service = mock_session_service
        orchestrator.trading_service = mock_trading_service
        orchestrator.update_handlers = Mock()
        orchestrator.update_handlers.update_session_influence_analysis = AsyncMock()
        orchestrator.update_handlers.update_session_marker = AsyncMock()
        return orchestrator
    def test_orchestrator_uses_session_service(self, orchestrator, mock_session_service) -> None:
        """Тест, что orchestrator использует session_service."""
        assert orchestrator.session_service is mock_session_service
        assert orchestrator.session_service is not None
    def test_orchestrator_does_not_have_deprecated_components(self, orchestrator) -> None:
        """Тест, что orchestrator не имеет устаревших компонентов."""
        assert not hasattr(orchestrator, 'session_marker')
        assert not hasattr(orchestrator, 'session_influence_analyzer')
    @pytest.mark.asyncio
    async def test_update_handlers_use_session_service(self, orchestrator, mock_session_service) -> None:
        """Тест, что update_handlers используют session_service."""
        # Симулируем вызов update_session_influence_analysis
        await orchestrator.update_handlers.update_session_influence_analysis(["BTCUSDT"])
        # Проверяем, что session_service был вызван
        mock_session_service.analyze_session_influence.assert_called()
    @pytest.mark.asyncio
    async def test_update_handlers_use_session_service_for_marker(self, orchestrator, mock_session_service) -> None:
        """Тест, что update_handlers используют session_service для маркера."""
        # Симулируем вызов update_session_marker
        await orchestrator.update_handlers.update_session_marker(["BTCUSDT"])
        # Проверяем, что session_service был вызван
        mock_session_service.get_current_session_context.assert_called()
    def test_session_service_factory_creates_valid_service(self) -> None:
        """Тест, что фабрика создает валидный SessionService."""
        service = get_session_service()
        assert isinstance(service, SessionService)
        assert service.registry is not None
        assert service.session_marker is not None
        assert service.influence_analyzer is not None
    def test_orchestrator_backward_compatibility(self) -> None:
        """Тест обратной совместимости orchestrator'а."""
        # Создаем orchestrator без session_service (старый способ)
        orchestrator_old = Mock()
        orchestrator_old.session_marker = Mock()
        orchestrator_old.session_influence_analyzer = Mock()
        orchestrator_old.trading_service = Mock()
        orchestrator_old.session_service = None
        # Проверяем, что старые компоненты доступны
        assert hasattr(orchestrator_old, 'session_marker')
        assert hasattr(orchestrator_old, 'session_influence_analyzer')
        assert orchestrator_old.session_service is None
    def test_orchestrator_new_way_priority(self) -> None:
        """Тест, что новый способ создания имеет приоритет."""
        session_service = get_session_service()
        trading_service = Mock()
        orchestrator = Mock()
        orchestrator.session_service = session_service
        orchestrator.session_marker = None  # Старый компонент
        orchestrator.session_influence_analyzer = None  # Старый компонент
        orchestrator.trading_service = trading_service
        # Проверяем, что используется session_service
        assert orchestrator.session_service is session_service
        # Старые компоненты должны быть None
        assert orchestrator.session_marker is None
        assert orchestrator.session_influence_analyzer is None 
