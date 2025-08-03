"""
Унифицированная система логирования для Infrastructure Layer.
"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from loguru import logger


class LogLevel(Enum):
    """Уровни логирования."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogContext(Enum):
    """Контексты логирования."""

    AGENT = "agent"
    REPOSITORY = "repository"
    SERVICE = "service"
    EXTERNAL = "external"
    CACHE = "cache"
    DATABASE = "database"
    NETWORK = "network"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class LogEntry:
    """Запись лога."""

    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel = LogLevel.INFO
    context: LogContext = LogContext.SERVICE
    module: str = ""
    operation: str = ""
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "context": self.context.value,
            "module": self.module,
            "operation": self.operation,
            "message": self.message,
            "data": self.data,
            "error": str(self.error) if self.error else None,
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
        }


class InfrastructureLogger:
    """Унифицированный логгер для Infrastructure Layer."""

    def __init__(
        self,
        name: str,
        context: LogContext = LogContext.SERVICE,
        enable_structured_logging: bool = True,
        enable_performance_logging: bool = True,
    ):
        self.name = name
        self.context = context
        self.enable_structured_logging = enable_structured_logging
        self.enable_performance_logging = enable_performance_logging
        self._logger = logger.bind(module=name, context=context.value)
        self._performance_metrics: Dict[str, float] = {}

    def trace(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне TRACE."""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне DEBUG."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне INFO."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне WARNING."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне ERROR."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне CRITICAL."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Внутренний метод логирования."""
        log_entry = LogEntry(
            level=level,
            context=self.context,
            module=self.name,
            message=message,
            data=kwargs,
        )
        if self.enable_structured_logging:
            self._logger.bind(**log_entry.to_dict()).log(level.value, message)
        else:
            self._logger.log(level.value, message)

    @contextmanager
    def operation_context(self, operation: str, **kwargs: Any) -> Any:
        """Контекстный менеджер для операций."""
        start_time = datetime.now()
        operation_logger = self._logger.bind(operation=operation, **kwargs)
        try:
            operation_logger.info(f"Starting operation: {operation}")
            yield operation_logger
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            operation_logger.info(
                f"Completed operation: {operation}", duration_ms=duration_ms
            )
            if self.enable_performance_logging:
                self._performance_metrics[operation] = duration_ms
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            operation_logger.error(
                f"Failed operation: {operation}", error=str(e), duration_ms=duration_ms
            )
            raise

    def log_performance(self, operation: str, duration_ms: float, **kwargs: Any) -> None:
        """Логирование производительности."""
        if self.enable_performance_logging:
            self._performance_metrics[operation] = duration_ms
            self.info(f"Performance: {operation}", duration_ms=duration_ms, **kwargs)

    def log_error(self, error: Exception, operation: str = "", **kwargs: Any) -> None:
        """Логирование ошибки."""
        self.error(
            f"Error in {operation or self.name}: {str(error)}",
            error_type=error.__class__.__name__,
            error_details=str(error),
            **kwargs,
        )

    def log_validation_error(self, field: str, value: Any, rule: str, **kwargs: Any) -> None:
        """Логирование ошибки валидации."""
        self.warning(
            f"Validation error: {field}",
            field=field,
            value=str(value),
            rule=rule,
            **kwargs,
        )

    def log_cache_hit(self, key: str, **kwargs: Any) -> None:
        """Логирование кэш-хита."""
        self.debug(f"Cache hit: {key}", cache_key=key, **kwargs)

    def log_cache_miss(self, key: str, **kwargs: Any) -> None:
        """Логирование кэш-промаха."""
        self.debug(f"Cache miss: {key}", cache_key=key, **kwargs)

    def log_external_request(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """Логирование внешнего запроса."""
        level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
        self._log(
            level,
            f"External request: {method} {endpoint}",
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Получение метрик производительности."""
        return self._performance_metrics.copy()

    def reset_performance_metrics(self) -> None:
        """Сброс метрик производительности."""
        self._performance_metrics.clear()


class LoggingConfig:
    """Конфигурация логирования."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        format_string: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = False,
        file_path: Optional[str] = None,
        enable_structured_logging: bool = True,
        enable_performance_logging: bool = True,
        max_file_size: str = "10 MB",
        rotation: str = "1 day",
        retention: str = "30 days",
    ):
        self.level = level
        self.format_string = format_string or self._get_default_format()
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.file_path = file_path
        self.enable_structured_logging = enable_structured_logging
        self.enable_performance_logging = enable_performance_logging
        self.max_file_size = max_file_size
        self.rotation = rotation
        self.retention = retention

    def _get_default_format(self) -> str:
        """Получение формата по умолчанию."""
        if self.enable_structured_logging:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level> | "
                "<blue>{extra}</blue>"
            )
        else:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )

    def configure_logging(self) -> None:
        """Настройка логирования."""
        # Удаляем стандартные обработчики
        logger.remove()
        # Добавляем консольный обработчик
        if self.enable_console:
            logger.add(
                sys.stdout,
                format=self.format_string,
                level=self.level.value,
                colorize=True,
            )
        # Добавляем файловый обработчик
        if self.enable_file and self.file_path:
            logger.add(
                self.file_path,
                format=self.format_string,
                level=self.level.value,
                rotation=self.rotation,
                retention=self.retention,
                compression="zip",
            )


# Глобальные утилиты
def get_logger(
    name: str, context: LogContext = LogContext.SERVICE, **kwargs: Any
) -> InfrastructureLogger:
    """Получение логгера."""
    return InfrastructureLogger(name, context, **kwargs)


def configure_infrastructure_logging(config: LoggingConfig) -> None:
    """Настройка логирования для infrastructure."""
    config.configure_logging()


from typing import TypeVar, Callable, Any, cast

F = TypeVar('F', bound=Callable[..., Any])

def log_performance_decorator(operation_name: str) -> Callable[[F], F]:
    """Декоратор для логирования производительности."""

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger_instance = get_logger(func.__module__)
            with logger_instance.operation_context(operation_name):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def log_errors_decorator() -> Any:
    """Декоратор для логирования ошибок."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger_instance = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger_instance.log_error(e, func.__name__)
                raise

        return wrapper

    return decorator


# Специализированные логгеры
class AgentLogger(InfrastructureLogger):
    """Логгер для агентов."""

    def __init__(self, agent_name: str, **kwargs):
        super().__init__(agent_name, LogContext.AGENT, **kwargs)

    def log_agent_state_change(self, old_state: str, new_state: str, **kwargs) -> None:
        """Логирование изменения состояния агента."""
        self.info(
            f"Agent state changed: {old_state} -> {new_state}",
            old_state=old_state,
            new_state=new_state,
            **kwargs,
        )

    def log_agent_processing(self, data_type: str, data_count: int, **kwargs) -> None:
        """Логирование обработки данных агентом."""
        self.debug(
            f"Processing {data_count} {data_type} items",
            data_type=data_type,
            data_count=data_count,
            **kwargs,
        )


class RepositoryLogger(InfrastructureLogger):
    """Логгер для репозиториев."""

    def __init__(self, repository_name: str, **kwargs):
        super().__init__(repository_name, LogContext.REPOSITORY, **kwargs)

    def log_database_operation(
        self,
        operation: str,
        entity_type: str,
        entity_count: int = 1,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Логирование операции с базой данных."""
        self.info(
            f"Database operation: {operation} {entity_count} {entity_type}",
            operation=operation,
            entity_type=entity_type,
            entity_count=entity_count,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_cache_operation(self, operation: str, key: str, **kwargs: Any) -> None:
        """Логирование операции с кэшем."""
        self.debug(
            f"Cache operation: {operation}",
            operation=operation,
            cache_key=key,
            **kwargs,
        )


class ServiceLogger(InfrastructureLogger):
    """Логгер для сервисов."""

    def __init__(self, service_name: str, **kwargs):
        super().__init__(service_name, LogContext.SERVICE, **kwargs)

    def log_service_call(
        self,
        method: str,
        parameters: Dict[str, Any],
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Логирование вызова сервиса."""
        self.info(
            f"Service call: {method}",
            method=method,
            parameters=parameters,
            duration_ms=duration_ms,
            **kwargs,
        )


class ExternalServiceLogger(InfrastructureLogger):
    """Логгер для внешних сервисов."""

    def __init__(self, service_name: str, **kwargs):
        super().__init__(service_name, LogContext.EXTERNAL, **kwargs)

    def log_api_request(
        self, method: str, url: str, status_code: int, duration_ms: float, **kwargs: Any
    ) -> None:
        """Логирование API запроса."""
        level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
        self._log(
            level,
            f"API request: {method} {url}",
            method=method,
            url=url,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )
