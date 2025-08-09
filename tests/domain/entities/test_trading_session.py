"""
Тесты для сущности TradingSession.
"""

import pytest
from datetime import datetime
from typing import Any
from uuid import UUID

from domain.entities.trading_session import TradingSession, SessionStatus


class TestTradingSession:
    """Тесты для сущности TradingSession."""

    @pytest.fixture
    def session(self) -> TradingSession:
        """Фикстура с валидной торговой сессией."""
        return TradingSession(
            id=UUID("550e8400-e29b-41d4-a716-446655440010"),
            portfolio_id=UUID("550e8400-e29b-41d4-a716-446655440011"),
            strategy_id=UUID("550e8400-e29b-41d4-a716-446655440012"),
            name="Test Trading Session",
            status=SessionStatus.ACTIVE,
            metadata={"user_id": "user_123", "exchange": "binance"},
        )

    def test_session_creation(self, session: TradingSession) -> None:
        """Тест создания торговой сессии."""
        assert str(session.id) == "550e8400-e29b-41d4-a716-446655440010"
        assert str(session.portfolio_id) == "550e8400-e29b-41d4-a716-446655440011"
        assert str(session.strategy_id) == "550e8400-e29b-41d4-a716-446655440012"
        assert session.name == "Test Trading Session"
        assert session.status == SessionStatus.ACTIVE
        assert session.metadata["user_id"] == "user_123"
        assert session.metadata["exchange"] == "binance"

    def test_session_default_values(self) -> None:
        """Тест значений по умолчанию."""
        session = TradingSession()
        assert session.id is not None
        assert session.portfolio_id is not None
        assert session.strategy_id is None
        assert session.name.startswith("Session_")
        assert session.status == SessionStatus.ACTIVE
        assert session.end_time is None
        assert session.metadata == {}

    def test_session_status_properties(self, session: TradingSession) -> None:
        """Тест свойств статуса сессии."""
        assert session.is_active is True
        assert session.is_paused is False
        assert session.is_closed is False
        assert session.is_error is False

        # Тест приостановленной сессии
        session.pause()
        assert session.is_active is False
        assert session.is_paused is True
        assert session.is_closed is False
        assert session.is_error is False

        # Тест закрытой сессии
        session.close()
        assert session.is_active is False
        assert session.is_paused is False
        assert session.is_closed is True
        assert session.is_error is False

        # Тест сессии в ошибке
        session.set_error()
        assert session.is_active is False
        assert session.is_paused is False
        assert session.is_closed is False
        assert session.is_error is True

    def test_session_operations(self, session: TradingSession) -> None:
        """Тест операций с сессией."""
        # Начальное состояние
        assert session.status == SessionStatus.ACTIVE

        # Приостановка
        session.pause()
        assert session.status == SessionStatus.PAUSED

        # Возобновление
        session.resume()
        assert session.status == SessionStatus.ACTIVE

        # Закрытие
        session.close()
        assert session.status == SessionStatus.CLOSED
        assert session.end_time is not None

        # Установка ошибки
        session.set_error()
        assert session.status == SessionStatus.ERROR

    def test_session_to_dict(self, session: TradingSession) -> None:
        """Тест преобразования в словарь."""
        data = session.to_dict()
        assert data["id"] == "550e8400-e29b-41d4-a716-446655440010"
        assert data["portfolio_id"] == "550e8400-e29b-41d4-a716-446655440011"
        assert data["strategy_id"] == "550e8400-e29b-41d4-a716-446655440012"
        assert data["name"] == "Test Trading Session"
        assert data["status"] == "active"
        assert data["metadata"]["user_id"] == "user_123"
        assert data["metadata"]["exchange"] == "binance"

    def test_session_from_dict(self, session: TradingSession) -> None:
        """Тест создания из словаря."""
        data = session.to_dict()
        restored_session = TradingSession.from_dict(data)

        assert str(restored_session.id) == str(session.id)
        assert str(restored_session.portfolio_id) == str(session.portfolio_id)
        assert str(restored_session.strategy_id) == str(session.strategy_id)
        assert restored_session.name == session.name
        assert restored_session.status == session.status
        assert restored_session.metadata == session.metadata

    def test_session_from_dict_with_defaults(self) -> None:
        """Тест создания из словаря с значениями по умолчанию."""
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440013",
            "portfolio_id": "550e8400-e29b-41d4-a716-446655440014",
            "status": "active",
            "start_time": {"value": "2024-01-01T12:00:00"},
        }
        session = TradingSession.from_dict(data)

        assert str(session.id) == "550e8400-e29b-41d4-a716-446655440013"
        assert str(session.portfolio_id) == "550e8400-e29b-41d4-a716-446655440014"
        assert session.strategy_id is None
        assert session.name.startswith("Session_")  # Имя автоматически генерируется
        assert session.status == SessionStatus.ACTIVE
        assert session.metadata == {}

    def test_session_string_representation(self, session: TradingSession) -> None:
        """Тест строкового представления."""
        str_repr = str(session)
        assert "TradingSession" in str_repr
        assert "Test Trading Session" in str_repr
        assert "active" in str_repr

    def test_session_repr_representation(self, session: TradingSession) -> None:
        """Тест представления для отладки."""
        repr_str = repr(session)
        assert "TradingSession" in repr_str
        assert "550e8400-e29b-41d4-a716-446655440010" in repr_str
        assert "550e8400-e29b-41d4-a716-446655440011" in repr_str
        assert "Test Trading Session" in repr_str
        assert "active" in repr_str

    def test_session_status_enum(self) -> None:
        """Тест перечисления статусов."""
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.CLOSED.value == "closed"
        assert SessionStatus.ERROR.value == "error"

    def test_session_without_strategy(self) -> None:
        """Тест сессии без стратегии."""
        session = TradingSession(
            id=UUID("550e8400-e29b-41d4-a716-446655440015"),
            portfolio_id=UUID("550e8400-e29b-41d4-a716-446655440016"),
            name="Session without strategy",
        )

        assert session.strategy_id is None
        assert session.name == "Session without strategy"
        assert session.status == SessionStatus.ACTIVE

    def test_session_close_sets_end_time(self, session: TradingSession) -> None:
        """Тест, что закрытие сессии устанавливает время окончания."""
        assert session.end_time is None
        session.close()
        assert session.end_time is not None
        assert session.status == SessionStatus.CLOSED

    def test_session_name_auto_generation(self) -> None:
        """Тест автоматической генерации имени сессии."""
        session = TradingSession()
        assert session.name.startswith("Session_")
        # Проверяем, что ID сессии содержится в имени (без дефисов)
        session_id_without_dashes = str(session.id).replace("-", "")
        assert session_id_without_dashes in session.name.replace("-", "")

    def test_session_metadata_persistence(self, session: TradingSession) -> None:
        """Тест сохранения метаданных при операциях."""
        original_metadata = session.metadata.copy()

        session.pause()
        assert session.metadata == original_metadata

        session.resume()
        assert session.metadata == original_metadata

        session.close()
        assert session.metadata == original_metadata

        session.set_error()
        assert session.metadata == original_metadata
