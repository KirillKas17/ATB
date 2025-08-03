"""
Протокол для базового сервиса.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseServiceProtocol(Protocol):
    """Протокол для базового сервиса."""

    def get_service_name(self) -> str:
        """Получение имени сервиса."""
        ...

    def get_service_version(self) -> str:
        """Получение версии сервиса."""
        ...

    def is_healthy(self) -> bool:
        """Проверка здоровья сервиса."""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик сервиса."""
        ...

    def initialize(self) -> None:
        """Инициализация сервиса."""
        ...

    def shutdown(self) -> None:
        """Завершение работы сервиса."""
        ...

    def get_last_error(self) -> Optional[str]:
        """Получение последней ошибки."""
        ...

    def get_start_time(self) -> datetime:
        """Получение времени запуска."""
        ...

    def get_uptime(self) -> float:
        """Получение времени работы в секундах."""
        ...


class BaseService(BaseServiceProtocol, ABC):
    """Базовая промышленная абстракция для сервисов домена."""

    def __init__(self) -> None:
        self._start_time = datetime.now()
        self._last_error: Optional[str] = None
        self._metrics: Dict[str, Any] = {}
        self._service_name = self.__class__.__name__
        self._service_version = "1.0.0"
        self._healthy = True

    def get_service_name(self) -> str:
        return self._service_name

    def get_service_version(self) -> str:
        return self._service_version

    def is_healthy(self) -> bool:
        return self._healthy

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def initialize(self) -> None:
        self._healthy = True
        self._start_time = datetime.now()
        self._last_error = None

    def shutdown(self) -> None:
        self._healthy = False

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def get_start_time(self) -> datetime:
        return self._start_time

    def get_uptime(self) -> float:
        return (datetime.now() - self._start_time).total_seconds()
