"""
Доменная сущность TradingSession.
Представляет торговую сессию.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from domain.value_objects.timestamp import Timestamp


class SessionStatus(Enum):
    """Статусы торговой сессии."""

    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class TradingSession:
    """
    Торговая сессия.
    Представляет собой активную торговую сессию с определенными
    параметрами и состоянием.
    """

    id: UUID = field(default_factory=uuid4)
    portfolio_id: UUID = field(default_factory=uuid4)
    strategy_id: Optional[UUID] = None
    name: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    start_time: Timestamp = field(default_factory=Timestamp.now)
    end_time: Optional[Timestamp] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Инициализация после создания объекта."""
        # Валидация обязательных полей
        if not self.name:
            self.name = f"Session_{self.id}"
        # Убеждаемся, что metadata является словарем
        # if not isinstance(self.metadata, dict):
        #     self.metadata = {}

    @property
    def is_active(self) -> bool:
        """Проверка, что сессия активна."""
        return self.status == SessionStatus.ACTIVE

    @property
    def is_paused(self) -> bool:
        """Проверка, что сессия приостановлена."""
        return self.status == SessionStatus.PAUSED

    @property
    def is_closed(self) -> bool:
        """Проверка, что сессия закрыта."""
        return self.status == SessionStatus.CLOSED

    @property
    def is_error(self) -> bool:
        """Проверка, что сессия в состоянии ошибки."""
        return self.status == SessionStatus.ERROR

    def pause(self) -> None:
        """Приостановка сессии."""
        self.status = SessionStatus.PAUSED

    def resume(self) -> None:
        """Возобновление сессии."""
        self.status = SessionStatus.ACTIVE

    def close(self) -> None:
        """Закрытие сессии."""
        self.status = SessionStatus.CLOSED
        self.end_time = Timestamp.now()

    def set_error(self) -> None:
        """Установка статуса ошибки."""
        self.status = SessionStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.
        Returns:
            Dict[str, Any]: Словарь с данными сессии
        """
        return {
            "id": str(self.id),
            "portfolio_id": str(self.portfolio_id),
            "strategy_id": str(self.strategy_id) if self.strategy_id else None,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.to_dict(),
            "end_time": self.end_time.to_dict() if self.end_time else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSession":
        """
        Создание из словаря.
        Args:
            data: Словарь с данными сессии
        Returns:
            TradingSession: Объект сессии
        """
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            portfolio_id=(
                UUID(data["portfolio_id"])
                if isinstance(data["portfolio_id"], str)
                else data["portfolio_id"]
            ),
            strategy_id=UUID(data["strategy_id"]) if data.get("strategy_id") else None,
            name=data.get("name", ""),
            status=SessionStatus(data["status"]),
            start_time=Timestamp.from_dict(data["start_time"]),
            end_time=(
                Timestamp.from_dict(data["end_time"]) if data.get("end_time") else None
            ),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """Строковое представление сессии."""
        return f"TradingSession({self.name}, status={self.status.value})"

    def __repr__(self) -> str:
        """Представление для отладки."""
        return (
            f"TradingSession(id={self.id}, portfolio_id={self.portfolio_id}, "
            f"name='{self.name}', status={self.status.value})"
        )
