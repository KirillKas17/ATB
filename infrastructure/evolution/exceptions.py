"""
Кастомные исключения для infrastructure/evolution слоя.
"""

from typing import Any, Dict, Optional


class EvolutionInfrastructureError(Exception):
    """Базовое исключение для infrastructure/evolution слоя."""

    def __init__(
        self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}


class StorageError(EvolutionInfrastructureError):
    """Исключение для ошибок хранилища."""

    def __init__(
        self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, f"storage_{error_type}", details)


class SerializationError(EvolutionInfrastructureError):
    """Исключение для ошибок сериализации/десериализации."""

    def __init__(
        self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, f"serialization_{error_type}", details)


class ValidationError(EvolutionInfrastructureError):
    """Исключение для ошибок валидации."""

    def __init__(self, message: str, field: str, value: Any) -> None:
        super().__init__(message, "validation_error", {"field": field, "value": value})


class ConnectionError(StorageError):
    """Исключение для ошибок подключения к БД."""

    def __init__(self, message: str, db_path: str) -> None:
        super().__init__(message, "connection_error", {"db_path": db_path})


class QueryError(StorageError):
    """Исключение для ошибок выполнения запросов."""

    def __init__(
        self, message: str, query: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, "query_error", {"query": query, "params": params})


class ConstraintError(StorageError):
    """Исключение для ошибок ограничений БД."""

    def __init__(self, message: str, constraint: str, table: str) -> None:
        super().__init__(
            message, "constraint_error", {"constraint": constraint, "table": table}
        )


class TimeoutError(StorageError):
    """Исключение для ошибок таймаута."""

    def __init__(self, message: str, timeout: float) -> None:
        super().__init__(message, "timeout_error", {"timeout": timeout})


class PermissionError(StorageError):
    """Исключение для ошибок прав доступа."""

    def __init__(self, message: str, operation: str, resource: str) -> None:
        super().__init__(
            message, "permission_error", {"operation": operation, "resource": resource}
        )


class CacheError(EvolutionInfrastructureError):
    """Исключение для ошибок кэша."""

    def __init__(
        self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, f"cache_{error_type}", details)


class BackupError(EvolutionInfrastructureError):
    """Исключение для ошибок резервного копирования."""

    def __init__(
        self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, f"backup_{error_type}", details)


class MigrationError(EvolutionInfrastructureError):
    """Исключение для ошибок миграций."""

    def __init__(
        self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, f"migration_{error_type}", details)
