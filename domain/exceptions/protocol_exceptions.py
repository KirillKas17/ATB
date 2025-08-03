"""
Специализированные исключения для промышленных протоколов.
Обеспечивают детальную обработку ошибок в протоколах домена.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Union
from uuid import UUID

from domain.types import (
    ModelId,
    OrderId,
    PortfolioId,
    PositionId,
    PredictionId,
    RiskProfileId,
    StrategyId,
    Symbol,
    TradeId,
)


class ProtocolError(Exception):
    """Базовое исключение для всех протоколов."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()


# ============================================================================
# EXCHANGE PROTOCOL EXCEPTIONS
# ============================================================================


class ExchangeConnectionError(ProtocolError):
    """Ошибка подключения к бирже."""

    def __init__(
        self,
        message: str,
        exchange_name: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.exchange_name = exchange_name


class ExchangeAuthenticationError(ProtocolError):
    """Ошибка аутентификации на бирже."""

    def __init__(
        self,
        message: str,
        exchange_name: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.exchange_name = exchange_name


class ExchangeRateLimitError(ProtocolError):
    """Ошибка превышения лимитов API биржи."""

    def __init__(
        self,
        message: str,
        exchange_name: str,
        retry_after: Optional[int] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.exchange_name = exchange_name
        self.retry_after = retry_after


class SymbolNotFoundError(ProtocolError):
    """Ошибка: символ не найден."""

    def __init__(
        self,
        message: str,
        symbol: Symbol,
        exchange_name: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.symbol = symbol
        self.exchange_name = exchange_name


class InvalidOrderError(ProtocolError):
    """Ошибка неверного ордера."""

    def __init__(
        self,
        message: str,
        order_id: Optional[OrderId] = None,
        symbol: Optional[Symbol] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.order_id = order_id
        self.symbol = symbol


class InsufficientBalanceError(ProtocolError):
    """Ошибка недостаточного баланса."""

    def __init__(
        self,
        message: str,
        required_amount: float,
        available_amount: float,
        currency: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.required_amount = required_amount
        self.available_amount = available_amount
        self.currency = currency


class OrderNotFoundError(ProtocolError):
    """Ошибка: ордер не найден."""

    def __init__(
        self,
        message: str,
        order_id: OrderId,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.order_id = order_id


class OrderAlreadyFilledError(ProtocolError):
    """Ошибка: ордер уже исполнен."""

    def __init__(
        self,
        message: str,
        order_id: OrderId,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.order_id = order_id


# ============================================================================
# ML PROTOCOL EXCEPTIONS
# ============================================================================


class ModelNotFoundError(ProtocolError):
    """Ошибка: модель не найдена."""

    def __init__(
        self,
        message: str,
        model_id: ModelId,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.model_id = model_id


class ModelNotReadyError(ProtocolError):
    """Ошибка: модель не готова."""

    def __init__(
        self,
        message: str,
        model_id: ModelId,
        current_status: str,
        required_status: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.model_id = model_id
        self.current_status = current_status
        self.required_status = required_status


class TrainingError(ProtocolError):
    """Ошибка обучения модели."""

    def __init__(
        self,
        message: str,
        model_id: ModelId,
        training_step: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.model_id = model_id
        self.training_step = training_step


class InsufficientDataError(ProtocolError):
    """Ошибка недостаточных данных."""

    def __init__(
        self,
        message: str,
        required_samples: int,
        available_samples: int,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.required_samples = required_samples
        self.available_samples = available_samples


class InvalidFeaturesError(ProtocolError):
    """Ошибка неверных признаков."""

    def __init__(
        self,
        message: str,
        model_id: ModelId,
        expected_features: list,
        provided_features: list,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.model_id = model_id
        self.expected_features = expected_features
        self.provided_features = provided_features


class ModelLoadError(ProtocolError):
    """Ошибка загрузки модели."""

    def __init__(
        self,
        message: str,
        model_id: ModelId,
        file_path: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.model_id = model_id
        self.file_path = file_path


# ============================================================================
# STRATEGY PROTOCOL EXCEPTIONS
# ============================================================================


class StrategyNotFoundError(ProtocolError):
    """Ошибка: стратегия не найдена."""

    def __init__(
        self,
        message: str,
        strategy_id: StrategyId,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.strategy_id = strategy_id


class StrategyExecutionError(ProtocolError):
    """Ошибка исполнения стратегии."""

    def __init__(
        self,
        message: str,
        strategy_id: StrategyId,
        execution_step: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.strategy_id = strategy_id
        self.execution_step = execution_step


class SignalGenerationError(ProtocolError):
    """Ошибка генерации сигнала."""

    def __init__(
        self,
        message: str,
        strategy_id: StrategyId,
        signal_type: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.strategy_id = strategy_id
        self.signal_type = signal_type


class InsufficientCapitalError(ProtocolError):
    """Ошибка недостаточного капитала."""

    def __init__(
        self,
        message: str,
        strategy_id: StrategyId,
        required_capital: float,
        available_capital: float,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.strategy_id = strategy_id
        self.required_capital = required_capital
        self.available_capital = available_capital


# ============================================================================
# REPOSITORY PROTOCOL EXCEPTIONS
# ============================================================================


class EntityNotFoundError(ProtocolError):
    """Ошибка: сущность не найдена."""

    def __init__(
        self,
        message: str,
        entity_type: str,
        entity_id: Union[UUID, str],
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.entity_type = entity_type
        self.entity_id = entity_id


class EntitySaveError(ProtocolError):
    """Ошибка сохранения сущности."""

    def __init__(
        self,
        message: str,
        entity_type: str,
        entity_id: Optional[Union[UUID, str]] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.entity_type = entity_type
        self.entity_id = entity_id


class EntityUpdateError(ProtocolError):
    """Ошибка обновления сущности."""

    def __init__(
        self, entity_type: str, entity_id: str, message: str = "Failed to update entity"
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{message}: {entity_type} with id {entity_id}")


class EntityDeleteError(ProtocolError):
    """Ошибка удаления сущности."""

    def __init__(
        self, entity_type: str, entity_id: str, message: str = "Failed to delete entity"
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{message}: {entity_type} with id {entity_id}")


class ProtocolConnectionError(ProtocolError):
    """Ошибка подключения к протоколу."""

    def __init__(self, service: str, message: str = "Connection failed"):
        super().__init__(message, "connection_error", {"service": service})
        self.service = service


class ConnectionError(ProtocolConnectionError):
    """Алиас для ConnectionError для совместимости."""

    pass


class ValidationError(ProtocolError):
    """Ошибка валидации."""

    def __init__(
        self,
        message: str,
        field_name: str,
        field_value: Any,
        validation_rule: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class DatabaseConnectionError(ProtocolError):
    """Ошибка соединения с базой данных."""

    def __init__(
        self,
        message: str,
        database_name: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.database_name = database_name


class TransactionError(ProtocolError):
    """Ошибка транзакции."""

    def __init__(
        self,
        message: str,
        transaction_id: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.transaction_id = transaction_id


# ============================================================================
# GENERAL PROTOCOL EXCEPTIONS
# ============================================================================


class TimeoutError(ProtocolError):
    """Ошибка таймаута."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class RetryExhaustedError(ProtocolError):
    """Ошибка исчерпания попыток повтора."""

    def __init__(
        self,
        message: str,
        max_retries: int,
        operation: str,
        last_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.max_retries = max_retries
        self.operation = operation
        self.last_error = last_error


class ConfigurationError(ProtocolError):
    """Ошибка конфигурации."""

    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Any,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.config_key = config_key
        self.config_value = config_value


class ServiceError(ProtocolError):
    """Ошибка сервиса."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.service_name = service_name


class ProtocolTimeoutError(ProtocolError):
    """Ошибка таймаута протокола."""

    def __init__(
        self, protocol: str, timeout: float, message: str = "Protocol timeout"
    ):
        self.protocol = protocol
        self.timeout = timeout
        super().__init__(f"{message}: {protocol} timed out after {timeout}s")


class ProtocolValidationError(ProtocolError):
    """Ошибка валидации протокола."""

    def __init__(
        self,
        protocol: str,
        field: str,
        value: Any,
        message: str = "Protocol validation failed",
    ):
        self.protocol = protocol
        self.field = field
        self.value = value
        super().__init__(f"{message}: {protocol} field {field} with value {value}")


class ProtocolRetryError(ProtocolError):
    """Ошибка повторных попыток протокола."""

    def __init__(
        self, protocol: str, attempts: int, message: str = "Protocol retry exhausted"
    ):
        self.protocol = protocol
        self.attempts = attempts
        super().__init__(f"{message}: {protocol} failed after {attempts} attempts")


class ProtocolCircuitBreakerError(ProtocolError):
    """Ошибка circuit breaker протокола."""

    def __init__(
        self, protocol: str, state: str, message: str = "Protocol circuit breaker open"
    ):
        self.protocol = protocol
        self.state = state
        super().__init__(f"{message}: {protocol} circuit breaker in {state} state")


class ProtocolRateLimitError(ProtocolError):
    """Ошибка rate limit протокола."""

    def __init__(
        self,
        protocol: str,
        limit: int,
        window: int,
        message: str = "Protocol rate limit exceeded",
    ):
        self.protocol = protocol
        self.limit = limit
        self.window = window
        super().__init__(f"{message}: {protocol} limit {limit} per {window}s")


class ProtocolHealthError(ProtocolError):
    """Ошибка здоровья протокола."""

    def __init__(
        self,
        protocol: str,
        health_status: str,
        message: str = "Protocol health check failed",
    ):
        self.protocol = protocol
        self.health_status = health_status
        super().__init__(f"{message}: {protocol} health status {health_status}")


class ProtocolPerformanceError(ProtocolError):
    """Ошибка производительности протокола."""

    def __init__(
        self,
        protocol: str,
        metric: str,
        value: float,
        threshold: float,
        message: str = "Protocol performance threshold exceeded",
    ):
        self.protocol = protocol
        self.metric = metric
        self.value = value
        self.threshold = threshold
        super().__init__(f"{message}: {protocol} {metric} {value} > {threshold}")


class ProtocolSecurityError(ProtocolError):
    """Ошибка безопасности протокола."""

    def __init__(
        self,
        protocol: str,
        security_issue: str,
        message: str = "Protocol security violation",
    ):
        self.protocol = protocol
        self.security_issue = security_issue
        super().__init__(f"{message}: {protocol} {security_issue}")


class ProtocolAuthenticationError(ProtocolError):
    """Ошибка аутентификации протокола."""

    def __init__(
        self,
        protocol: str,
        auth_issue: str,
        message: str = "Protocol authentication failed",
    ):
        self.protocol = protocol
        self.auth_issue = auth_issue
        super().__init__(f"{message}: {protocol} {auth_issue}")


class ProtocolAuthorizationError(ProtocolError):
    """Ошибка авторизации протокола."""

    def __init__(
        self,
        protocol: str,
        auth_issue: str,
        message: str = "Protocol authorization failed",
    ):
        self.protocol = protocol
        self.auth_issue = auth_issue
        super().__init__(f"{message}: {protocol} {auth_issue}")


class ProtocolIntegrityError(ProtocolError):
    """Ошибка целостности протокола."""

    def __init__(
        self,
        protocol: str,
        integrity_issue: str,
        message: str = "Protocol integrity violation",
    ):
        self.protocol = protocol
        self.integrity_issue = integrity_issue
        super().__init__(f"{message}: {protocol} {integrity_issue}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base
    "ProtocolError",
    # Exchange exceptions
    "ExchangeConnectionError",
    "ExchangeAuthenticationError",
    "ExchangeRateLimitError",
    "SymbolNotFoundError",
    "InvalidOrderError",
    "InsufficientBalanceError",
    "OrderNotFoundError",
    "OrderAlreadyFilledError",
    # ML exceptions
    "ModelNotFoundError",
    "ModelNotReadyError",
    "TrainingError",
    "InsufficientDataError",
    "InvalidFeaturesError",
    "ModelLoadError",
    # Strategy exceptions
    "StrategyNotFoundError",
    "StrategyExecutionError",
    "SignalGenerationError",
    "InsufficientCapitalError",
    # Repository exceptions
    "EntityNotFoundError",
    "EntitySaveError",
    "EntityUpdateError",
    "EntityDeleteError",
    "ValidationError",
    "DatabaseConnectionError",
    "TransactionError",
    # General exceptions
    "TimeoutError",
    "RetryExhaustedError",
    "ConfigurationError",
    "ConnectionError",
    "ProtocolTimeoutError",
    "ProtocolValidationError",
    "ProtocolRetryError",
    "ProtocolCircuitBreakerError",
    "ProtocolRateLimitError",
    "ProtocolHealthError",
    "ProtocolPerformanceError",
    "ProtocolSecurityError",
    "ProtocolAuthenticationError",
    "ProtocolAuthorizationError",
    "ProtocolIntegrityError",
]
