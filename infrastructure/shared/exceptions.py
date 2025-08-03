"""
Унифицированные исключения для Infrastructure Layer.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Union


class InfrastructureError(Exception):
    """Базовое исключение для Infrastructure Layer."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для логирования."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class RepositoryError(InfrastructureError):
    """Ошибка репозитория."""

    pass


class CacheError(InfrastructureError):
    """Ошибка кэширования."""

    pass


class ExternalServiceError(InfrastructureError):
    """Ошибка внешнего сервиса."""

    def __init__(
        self,
        message: str,
        service_name: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_data = response_data or {}

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "service_name": self.service_name,
                "endpoint": self.endpoint,
                "status_code": self.status_code,
                "response_data": self.response_data,
            }
        )
        return base_dict


class AgentError(InfrastructureError):
    """Ошибка агента."""

    def __init__(
        self, message: str, agent_name: str, agent_type: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.agent_name = agent_name
        self.agent_type = agent_type

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({"agent_name": self.agent_name, "agent_type": self.agent_type})
        return base_dict


class ServiceError(InfrastructureError):
    """Ошибка сервиса."""

    def __init__(
        self, message: str, service_name: str, operation: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {"service_name": self.service_name, "operation": self.operation}
        )
        return base_dict


class ValidationError(InfrastructureError):
    """Ошибка валидации."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rules = validation_rules or {}

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "field_name": self.field_name,
                "field_value": (
                    str(self.field_value) if self.field_value is not None else None
                ),
                "validation_rules": self.validation_rules,
            }
        )
        return base_dict


class ConfigurationError(InfrastructureError):
    """Ошибка конфигурации."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "config_key": self.config_key,
                "config_value": (
                    str(self.config_value) if self.config_value is not None else None
                ),
            }
        )
        return base_dict


class InfrastructureConnectionError(InfrastructureError):
    """Ошибка соединения в инфраструктуре."""

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port
        self.timeout = timeout

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {"host": self.host, "port": self.port, "timeout": self.timeout}
        )
        return base_dict


class InfrastructureTimeoutError(InfrastructureError):
    """Ошибка таймаута в инфраструктуре."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {"timeout_seconds": self.timeout_seconds, "operation": self.operation}
        )
        return base_dict


class ResourceError(InfrastructureError):
    """Ошибка ресурсов."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {"resource_type": self.resource_type, "resource_id": self.resource_id}
        )
        return base_dict


class DataIntegrityError(InfrastructureError):
    """Ошибка целостности данных."""

    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        constraint_violation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.constraint_violation = constraint_violation

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "entity_type": self.entity_type,
                "entity_id": self.entity_id,
                "constraint_violation": self.constraint_violation,
            }
        )
        return base_dict


class PerformanceError(InfrastructureError):
    """Ошибка производительности."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.operation = operation
        self.execution_time_ms = execution_time_ms
        self.threshold_ms = threshold_ms

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "operation": self.operation,
                "execution_time_ms": self.execution_time_ms,
                "threshold_ms": self.threshold_ms,
            }
        )
        return base_dict


# Специфичные исключения для разных модулей


class MarketDataError(InfrastructureError):
    """Ошибка рыночных данных."""

    pass


class TradingError(InfrastructureError):
    """Ошибка торговых операций."""

    pass


class MLServiceError(InfrastructureError):
    """Ошибка ML сервиса."""

    pass


class MessagingError(InfrastructureError):
    """Ошибка обмена сообщениями."""

    pass


class EvolutionError(InfrastructureError):
    """Ошибка эволюционных алгоритмов."""

    pass


class EntitySystemError(InfrastructureError):
    """Ошибка системы сущностей."""

    pass


# Утилиты для работы с исключениями


def create_error_context(module: str, operation: str, **kwargs: Any) -> Dict[str, Any]:
    """Создание контекста ошибки."""
    return {
        "module": module,
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }


def format_error_message(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> str:
    """Форматирование сообщения об ошибке."""
    if isinstance(error, InfrastructureError):
        error_dict = error.to_dict()
        if context:
            error_dict["context"] = context
        return f"{error_dict['error_type']}: {error_dict['message']}"
    else:
        return f"{error.__class__.__name__}: {str(error)}"


def is_infrastructure_error(error: Exception) -> bool:
    """Проверка является ли ошибка инфраструктурной."""
    return isinstance(error, InfrastructureError)


def get_error_severity(error: Exception) -> str:
    """Получение уровня серьезности ошибки."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return "critical"
    elif isinstance(error, (ValidationError, ConfigurationError)):
        return "warning"
    elif isinstance(error, (PerformanceError, ResourceError)):
        return "error"
    else:
        return "info"
