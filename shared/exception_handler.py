"""
Улучшенный обработчик исключений для безопасной обработки ошибок.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union


class ExceptionSeverity(Enum):
    """Уровни серьезности исключений."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExceptionCategory(Enum):
    """Категории исключений."""

    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    UNKNOWN = "unknown"


@dataclass
class ExceptionContext:
    """Контекст исключения."""

    severity: ExceptionSeverity
    category: ExceptionCategory
    operation: str
    component: str
    details: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))


class SafeExceptionHandler:
    """Безопасный обработчик исключений."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._exception_handlers: Dict[Type[Exception], Callable] = {}
        self._default_handler: Optional[Callable] = None

    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception, ExceptionContext], Any],
    ) -> None:
        """Регистрация обработчика для конкретного типа исключения."""
        self._exception_handlers[exception_type] = handler

    def set_default_handler(
        self, handler: Callable[[Exception, ExceptionContext], Any]
    ) -> None:
        """Установка обработчика по умолчанию."""
        self._default_handler = handler

    def handle_exception(self, exception: Exception, context: ExceptionContext) -> Any:
        """Обработка исключения с контекстом."""
        try:
            # Ищем специфичный обработчик
            for exc_type, handler in self._exception_handlers.items():
                if isinstance(exception, exc_type):
                    return handler(exception, context)
            # Используем обработчик по умолчанию
            if self._default_handler:
                return self._default_handler(exception, context)
            # Логируем исключение
            self._log_exception(exception, context)
            return None
        except Exception as handler_error:
            # Если обработчик сам упал, логируем это
            self.logger.error(f"Exception handler failed: {handler_error}")
            self._log_exception(exception, context)
            return None

    def _log_exception(self, exception: Exception, context: ExceptionContext) -> None:
        """Логирование исключения с контекстом."""
        log_message = (
            f"Exception in {context.component}.{context.operation}: "
            f"{type(exception).__name__}: {str(exception)}"
        )
        if context.severity == ExceptionSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=True, extra=context.details)
        elif context.severity == ExceptionSeverity.HIGH:
            self.logger.error(log_message, exc_info=True, extra=context.details)
        elif context.severity == ExceptionSeverity.MEDIUM:
            self.logger.warning(log_message, exc_info=False, extra=context.details)
        else:
            self.logger.info(log_message, exc_info=False, extra=context.details)


def handle_exceptions(
    component: str,
    operation: str,
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    category: ExceptionCategory = ExceptionCategory.UNKNOWN,
    reraise: bool = False,
    default_return: Any = None,
) -> Callable[[Callable], Callable]:
    """Декоратор для безопасной обработки исключений."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ExceptionContext(
                    severity=severity,
                    category=category,
                    operation=operation,
                    component=component,
                    details={
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "function": func.__name__,
                    },
                    timestamp=datetime.now(),
                )
                # Используем глобальный обработчик
                handler = get_global_exception_handler()
                result = handler.handle_exception(e, context)
                if reraise:
                    raise e
                return result if result is not None else default_return

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ExceptionContext(
                    severity=severity,
                    category=category,
                    operation=operation,
                    component=component,
                    details={
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "function": func.__name__,
                    },
                    timestamp=datetime.now(),
                )
                # Используем глобальный обработчик
                handler = get_global_exception_handler()
                result = handler.handle_exception(e, context)
                if reraise:
                    raise e
                return result if result is not None else default_return

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


def safe_execute(
    operation: Callable[..., Any],
    *args: Any,
    component: str = "unknown",
    operation_name: str = "unknown",
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    category: ExceptionCategory = ExceptionCategory.UNKNOWN,
    default_return: Any = None,
    **kwargs: Any,
) -> Any:
    """Безопасное выполнение операции."""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        context = ExceptionContext(
            severity=severity,
            category=category,
            operation=operation_name,
            component=component,
            details={
                "args": str(args),
                "kwargs": str(kwargs),
                "operation": operation.__name__,
            },
            timestamp=datetime.now(),
        )
        handler = get_global_exception_handler()
        result = handler.handle_exception(e, context)
        return result if result is not None else default_return


async def safe_execute_async(
    operation: Callable[..., Any],
    *args: Any,
    component: str = "unknown",
    operation_name: str = "unknown",
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    category: ExceptionCategory = ExceptionCategory.UNKNOWN,
    default_return: Any = None,
    **kwargs: Any,
) -> Any:
    """Безопасное выполнение асинхронной операции."""
    try:
        return await operation(*args, **kwargs)
    except Exception as e:
        context = ExceptionContext(
            severity=severity,
            category=category,
            operation=operation_name,
            component=component,
            details={
                "args": str(args),
                "kwargs": str(kwargs),
                "operation": operation.__name__,
            },
            timestamp=datetime.now(),
        )
        handler = get_global_exception_handler()
        result = handler.handle_exception(e, context)
        return result if result is not None else default_return


# Глобальный обработчик исключений
_global_handler: Optional[SafeExceptionHandler] = None


def get_global_exception_handler() -> SafeExceptionHandler:
    """Получение глобального обработчика исключений."""
    global _global_handler
    if _global_handler is None:
        _global_handler = SafeExceptionHandler()
        _setup_default_handlers(_global_handler)
    return _global_handler


def set_global_exception_handler(handler: SafeExceptionHandler) -> None:
    """Установка глобального обработчика исключений."""
    global _global_handler
    _global_handler = handler


def _setup_default_handlers(handler: SafeExceptionHandler) -> None:
    """Настройка обработчиков по умолчанию."""

    # Обработчик для ValidationError
    def handle_validation_error(
        exception: Exception, context: ExceptionContext
    ) -> Optional[Any]:
        context.logger.warning(f"Validation error: {exception}")
        return None

    # Обработчик для ConnectionError
    def handle_connection_error(
        exception: Exception, context: ExceptionContext
    ) -> Optional[Any]:
        context.logger.error(f"Connection error: {exception}")
        return None

    # Обработчик для TimeoutError
    def handle_timeout_error(exception: Exception, context: ExceptionContext) -> Optional[Any]:
        context.logger.error(f"Timeout error: {exception}")
        return None

    # Обработчик по умолчанию
    def default_handler(exception: Exception, context: ExceptionContext) -> Optional[Any]:
        context.logger.error(f"Unhandled exception: {exception}")
        return None

    # Регистрируем обработчики
    try:
        from domain.exceptions import ValidationError

        handler.register_handler(ValidationError, handle_validation_error)
    except ImportError:
        pass
    handler.register_handler(ConnectionError, handle_connection_error)
    handler.register_handler(TimeoutError, handle_timeout_error)
    handler.set_default_handler(default_handler)



