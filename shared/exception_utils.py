"""
Утилиты для обработки исключений и устранения дублирования try/except блоков.
"""

import asyncio
import functools
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from loguru import logger

from domain.exceptions.base_exceptions import (
    DomainException,
    RepositoryError,
    EvaluationError,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ExceptionHandler:
    """Обработчик исключений с контекстом."""

    def __init__(self, context: str = "") -> None:
        """Инициализация обработчика."""
        self.context = context
        self.logger = logger.bind(context=context)

    def handle(
        self,
        error: Exception,
        operation: str = "",
        default_value: Any = None,
        reraise: bool = True,
    ) -> Any:
        """Обработка исключения."""
        error_msg = (
            f"Operation '{operation}' failed in context '{self.context}': {str(error)}"
        )
        self.logger.error(error_msg, exc_info=True)

        if reraise:
            raise error

        return default_value

    async def handle_async(
        self,
        error: Exception,
        operation: str = "",
        default_value: Any = None,
        reraise: bool = True,
    ) -> Any:
        """Асинхронная обработка исключения."""
        return self.handle(error, operation, default_value, reraise)


class ExceptionUtils:
    """Утилиты для работы с исключениями."""

    # ============================================================================
    # ДЕКОРАТОРЫ ДЛЯ ОБРАБОТКИ ИСКЛЮЧЕНИЙ
    # ============================================================================

    @staticmethod
    def handle_exceptions(
        context: str = "",
        default_value: Any = None,
        reraise: bool = True,
        log_errors: bool = True,
        specific_exceptions: Optional[List[Type[Exception]]] = None,
    ) -> Callable[[F], Any]:
        """
        Декоратор для обработки исключений.

        Args:
            context: Контекст операции
            default_value: Значение по умолчанию при ошибке
            reraise: Перебрасывать ли исключение
            log_errors: Логировать ли ошибки
            specific_exceptions: Список конкретных исключений для обработки
        """

        def decorator(func: F) -> Any:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> None:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Проверка на конкретные исключения
                    if specific_exceptions and not any(
                        isinstance(e, exc_type) for exc_type in specific_exceptions
                    ):
                        raise

                    if log_errors:
                        logger.error(
                            f"Function '{func.__name__}' failed in context '{context}': {str(e)}",
                            exc_info=True,
                        )

                    if reraise:
                        raise

                    return default_value

            return wrapper

        return decorator

    @staticmethod
    def handle_async_exceptions(
        context: str = "",
        default_value: Any = None,
        reraise: bool = True,
        log_errors: bool = True,
        specific_exceptions: Optional[List[Type[Exception]]] = None,
    ) -> Callable[[F], Any]:
        """
        Декоратор для обработки исключений в асинхронных функциях.

        Args:
            context: Контекст операции
            default_value: Значение по умолчанию при ошибке
            reraise: Перебрасывать ли исключение
            log_errors: Логировать ли ошибки
            specific_exceptions: Список конкретных исключений для обработки
        """

        def decorator(func: F) -> Any:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> None:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Проверка на конкретные исключения
                    if specific_exceptions and not any(
                        isinstance(e, exc_type) for exc_type in specific_exceptions
                    ):
                        raise

                    if log_errors:
                        logger.error(
                            f"Async function '{func.__name__}' failed in context '{context}': {str(e)}",
                            exc_info=True,
                        )

                    if reraise:
                        raise

                    return default_value

            return wrapper

        return decorator

    # ============================================================================
    # КОНТЕКСТНЫЕ МЕНЕДЖЕРЫ
    # ============================================================================

    @staticmethod
    @contextmanager
    def exception_context(
        context: str = "",
        default_value: Any = None,
        reraise: bool = True,
        log_errors: bool = True,
    ):
        """
        Контекстный менеджер для обработки исключений.

        Args:
            context: Контекст операции
            default_value: Значение по умолчанию при ошибке
            reraise: Перебрасывать ли исключение
            log_errors: Логировать ли ошибки
        """
        try:
            yield
        except Exception as e:
            if log_errors:
                logger.error(
                    f"Operation failed in context '{context}': {str(e)}", exc_info=True
                )

            if reraise:
                raise

            return default_value

    @staticmethod
    @asynccontextmanager
    async def async_exception_context(
        context: str = "",
        default_value: Any = None,
        reraise: bool = True,
        log_errors: bool = True,
    ):
        """
        Асинхронный контекстный менеджер для обработки исключений.

        Args:
            context: Контекст операции
            default_value: Значение по умолчанию при ошибке
            reraise: Перебрасывать ли исключение
            log_errors: Логировать ли ошибки
        """
        try:
            yield
        except Exception as e:
            if log_errors:
                logger.error(
                    f"Async operation failed in context '{context}': {str(e)}",
                    exc_info=True,
                )

            if reraise:
                raise

            # Не используем return в async generator
            # return default_value

    # ============================================================================
    # УТИЛИТЫ ДЛЯ ОБРАБОТКИ СПЕЦИФИЧНЫХ ИСКЛЮЧЕНИЙ
    # ============================================================================

    @staticmethod
    def handle_validation_errors(func: F) -> F:
        """Декоратор для обработки ошибок валидации."""
        return ExceptionUtils.handle_exceptions(
            context="validation", reraise=True, specific_exceptions=[EvaluationError]
        )(func)

    @staticmethod
    def handle_service_errors(func: F) -> F:
        """Декоратор для обработки ошибок сервисов."""
        return ExceptionUtils.handle_exceptions(
            context="service", reraise=True, specific_exceptions=[DomainException]
        )(func)

    @staticmethod
    def handle_repository_errors(func: F) -> F:
        """Декоратор для обработки ошибок репозиториев."""
        return ExceptionUtils.handle_exceptions(
            context="repository", reraise=True, specific_exceptions=[RepositoryError]
        )(func)

    @staticmethod
    def handle_network_errors(func: F) -> F:
        """Декоратор для обработки сетевых ошибок."""
        network_exceptions: List[Type[Exception]] = [
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError,
            # asyncio.CancelledError убран, так как это BaseException
        ]
        return ExceptionUtils.handle_exceptions(
            context="network", reraise=False, specific_exceptions=network_exceptions
        )(func)

    # ============================================================================
    # УТИЛИТЫ ДЛЯ ПОВТОРНЫХ ПОПЫТОК
    # ============================================================================

    @staticmethod
    def retry_on_exception(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Optional[List[Type[Exception]]] = None,
        context: str = "",
    ) -> Callable[[F], Any]:
        """
        Декоратор для повторных попыток при исключениях.

        Args:
            max_attempts: Максимальное количество попыток
            delay: Начальная задержка в секундах
            backoff_factor: Множитель задержки
            exceptions: Список исключений для повторных попыток
            context: Контекст операции
        """

        def decorator(func: F) -> Any:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> None:
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        # Проверка на конкретные исключения
                        if exceptions and not any(
                            isinstance(e, exc_type) for exc_type in exceptions
                        ):
                            raise

                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed in context '{context}': {str(e)}. "
                                f"Retrying in {current_delay} seconds..."
                            )
                            asyncio.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(
                                f"All {max_attempts} attempts failed in context '{context}': {str(e)}"
                            )
                            raise last_exception

                raise last_exception

            return wrapper

        return decorator

    @staticmethod
    def retry_async_on_exception(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Optional[List[Type[Exception]]] = None,
        context: str = "",
    ) -> Callable[[F], Any]:
        """
        Декоратор для повторных попыток в асинхронных функциях.

        Args:
            max_attempts: Максимальное количество попыток
            delay: Начальная задержка в секундах
            backoff_factor: Множитель задержки
            exceptions: Список исключений для повторных попыток
            context: Контекст операции
        """

        def decorator(func: F) -> Any:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> None:
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        # Проверка на конкретные исключения
                        if exceptions and not any(
                            isinstance(e, exc_type) for exc_type in exceptions
                        ):
                            raise

                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed in context '{context}': {str(e)}. "
                                f"Retrying in {current_delay} seconds..."
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(
                                f"All {max_attempts} attempts failed in context '{context}': {str(e)}"
                            )
                            raise last_exception

                raise last_exception

            return wrapper

        return decorator

    # ============================================================================
    # УТИЛИТЫ ДЛЯ АНАЛИЗА ИСКЛЮЧЕНИЙ
    # ============================================================================

    @staticmethod
    def get_exception_info(error: Exception) -> Dict[str, Any]:
        """Получение информации об исключении."""
        return {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "traceback": ExceptionUtils._get_traceback(error),
        }

    @staticmethod
    def _get_traceback(error: Exception) -> List[str]:
        """Получение трейсбека исключения."""
        import traceback

        return traceback.format_exception(type(error), error, error.__traceback__)

    @staticmethod
    def is_recoverable_error(error: Exception) -> bool:
        """Проверка, является ли ошибка восстанавливаемой."""
        recoverable_types = [
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError,
            asyncio.CancelledError,
            ValueError,
            TypeError,
        ]
        return any(isinstance(error, exc_type) for exc_type in recoverable_types)

    @staticmethod
    def is_critical_error(error: Exception) -> bool:
        """Проверка, является ли ошибка критической."""
        critical_types = [MemoryError, SystemError, KeyboardInterrupt, BaseException]
        return any(isinstance(error, exc_type) for exc_type in critical_types)

    # ============================================================================
    # УТИЛИТЫ ДЛЯ ЛОГИРОВАНИЯ
    # ============================================================================

    @staticmethod
    def log_exception(
        error: Exception, context: str = "", level: str = "ERROR"
    ) -> None:
        """Логирование исключения с контекстом."""
        log_func = getattr(logger, level.lower())
        log_func(
            f"Exception in context '{context}': {type(error).__name__}: {str(error)}",
            exc_info=True,
        )

    @staticmethod
    def log_exception_with_metrics(
        error: Exception, context: str = "", metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Логирование исключения с метриками."""
        error_info = ExceptionUtils.get_exception_info(error)
        if metrics:
            error_info["metrics"] = metrics

        logger.error(f"Exception in context '{context}': {error_info}", exc_info=True)
