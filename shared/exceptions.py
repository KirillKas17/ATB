"""
Централизованная система ошибок для проекта Syntra.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorSeverity(Enum):
    """Уровни серьезности ошибок."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Категории ошибок."""

    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_SERVICE = "external_service"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Контекст ошибки."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntraException(Exception):
    """Базовый класс для всех исключений Syntra."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.details = details or {}
        self.retryable = retryable
        self.timestamp = datetime.utcnow()

    def _generate_error_code(self) -> str:
        """Генерация кода ошибки."""
        return f"{self.category.value.upper()}_{self.__class__.__name__.upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "retryable": self.retryable,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "component": self.context.component,
                "operation": self.context.operation,
                "metadata": self.context.metadata,
            },
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


# Алиас для обратной совместимости
BaseException = SyntraException


# Ошибки обработки
class ProcessingError(SyntraException):
    """Ошибка обработки данных."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


# Ошибки валидации
class ValidationError(SyntraException):
    """Ошибка валидации."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class BusinessLogicError(SyntraException):
    """Ошибка бизнес-логики."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class InsufficientFundsError(BusinessLogicError):
    """Недостаточно средств."""

    def __init__(
        self, required: float, available: float, currency: str = "USDT", **kwargs: Any
    ) -> None:
        message = f"Insufficient funds. Required: {required} {currency}, Available: {available} {currency}"
        super().__init__(message, **kwargs)
        self.details.update(
            {"required": required, "available": available, "currency": currency}
        )


class InvalidOrderError(BusinessLogicError):
    """Некорректный ордер."""

    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if order_id:
            self.details["order_id"] = order_id


class InvalidPositionError(BusinessLogicError):
    """Некорректная позиция."""

    def __init__(self, message: str, position_id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if position_id:
            self.details["position_id"] = position_id


# Ошибки инфраструктуры
class InfrastructureError(SyntraException):
    """Ошибка инфраструктуры."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class DatabaseError(InfrastructureError):
    """Ошибка базы данных."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if operation:
            self.details["operation"] = operation


class CacheError(InfrastructureError):
    """Ошибка кэша."""

    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if cache_key:
            self.details["cache_key"] = cache_key


class MessageQueueError(InfrastructureError):
    """Ошибка очереди сообщений."""

    def __init__(self, message: str, queue_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if queue_name:
            self.details["queue_name"] = queue_name


# Ошибки внешних сервисов
class ExternalServiceError(SyntraException):
    """Ошибка внешнего сервиса."""

    def __init__(
        self,
        message: str,
        service_name: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            retryable=True,
            **kwargs,
        )
        self.details.update(
            {
                "service_name": service_name,
                "endpoint": endpoint,
                "status_code": status_code,
            }
        )


class ExchangeError(ExternalServiceError):
    """Ошибка биржи."""

    def __init__(self, message: str, exchange_name: str, **kwargs: Any) -> None:
        super().__init__(message, service_name=exchange_name, **kwargs)


class APIError(ExternalServiceError):
    """Ошибка API."""

    def __init__(self, message: str, api_name: str, **kwargs: Any) -> None:
        super().__init__(message, service_name=api_name, **kwargs)


# Ошибки безопасности
class SecurityError(SyntraException):
    """Ошибка безопасности."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


class AuthenticationError(SecurityError):
    """Ошибка аутентификации."""

    def __init__(self, message: str, user_id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if user_id:
            self.details["user_id"] = user_id


class AuthorizationError(SecurityError):
    """Ошибка авторизации."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if user_id:
            self.details["user_id"] = user_id
        if resource:
            self.details["resource"] = resource


# Ошибки производительности
class PerformanceError(SyntraException):
    """Ошибка производительности."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class TimeoutError(PerformanceError):
    """Ошибка таймаута."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class RateLimitError(PerformanceError):
    """Ошибка лимита запросов."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if retry_after:
            self.details["retry_after"] = retry_after


# Ошибки целостности данных
class DataIntegrityError(SyntraException):
    """Ошибка целостности данных."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_INTEGRITY,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class DataCorruptionError(DataIntegrityError):
    """Ошибка повреждения данных."""

    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if data_type:
            self.details["data_type"] = data_type


class DataInconsistencyError(DataIntegrityError):
    """Ошибка несогласованности данных."""

    def __init__(
        self, message: str, conflicting_fields: Optional[List[str]] = None, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        if conflicting_fields:
            self.details["conflicting_fields"] = conflicting_fields


# Системные ошибки
class SystemError(SyntraException):
    """Системная ошибка."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


class ConfigurationError(SystemError):
    """Ошибка конфигурации."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if config_key:
            self.details["config_key"] = config_key


class InitializationError(SystemError):
    """Ошибка инициализации."""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if component:
            self.details["component"] = component


# Функции для работы с ошибками
def create_error_context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    **metadata: Any,
) -> ErrorContext:
    """Создание контекста ошибки."""
    return ErrorContext(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        component=component,
        operation=operation,
        metadata=metadata,
    )


def is_retryable_error(error: SyntraException) -> bool:
    """Проверка возможности повторной попытки."""
    return error.retryable


def get_error_severity(error: SyntraException) -> ErrorSeverity:
    """Получение серьезности ошибки."""
    return error.severity


def get_error_category(error: SyntraException) -> ErrorCategory:
    """Получение категории ошибки."""
    return error.category


def format_error_for_logging(error: SyntraException) -> Dict[str, Any]:
    """Форматирование ошибки для логирования."""
    return {
        "error_code": error.error_code,
        "message": error.message,
        "severity": error.severity.value,
        "category": error.category.value,
        "retryable": error.retryable,
        "timestamp": error.timestamp.isoformat(),
        "context": error.context.__dict__,
        "details": error.details,
        "traceback": str(error),
    }


def create_validation_error(
    message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs: Any
) -> ValidationError:
    """Создание ошибки валидации."""
    return ValidationError(message, field, value, **kwargs)


def create_business_logic_error(message: str, **kwargs: Any) -> BusinessLogicError:
    """Создание ошибки бизнес-логики."""
    return BusinessLogicError(message, **kwargs)


def create_infrastructure_error(message: str, **kwargs: Any) -> InfrastructureError:
    """Создание ошибки инфраструктуры."""
    return InfrastructureError(message, **kwargs)


def create_external_service_error(
    message: str, service_name: str, **kwargs: Any
) -> ExternalServiceError:
    """Создание ошибки внешнего сервиса."""
    return ExternalServiceError(message, service_name, **kwargs)


def create_security_error(message: str, **kwargs: Any) -> SecurityError:
    """Создание ошибки безопасности."""
    return SecurityError(message, **kwargs)
