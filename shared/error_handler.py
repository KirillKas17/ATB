"""
Централизованный обработчик ошибок для Syntra.
"""

import asyncio
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

from .exceptions import ConfigurationError, ExchangeError
from infrastructure.shared.exceptions import TradingError
from .logging import get_logger

logger = get_logger(__name__)


class ErrorHandler:
    """Централизованный обработчик ошибок."""

    def __init__(self) -> None:
        self.error_callbacks: Dict[Type[Exception], Callable] = {}
        self.error_counters: Dict[str, int] = {}
        self.error_history: list = []
        self.max_history_size = 1000

    def register_error_callback(self, error_type: Type[Exception], callback: Callable) -> None:
        """Регистрация callback для определенного типа ошибки."""
        self.error_callbacks[error_type] = callback

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Обработка ошибки.

        Args:
            error: Исключение для обработки
            context: Дополнительный контекст

        Returns:
            bool: True если ошибка была обработана, False если нужно пробросить дальше
        """
        try:
            # Увеличиваем счетчик ошибок
            error_type = type(error).__name__
            self.error_counters[error_type] = self.error_counters.get(error_type, 0) + 1

            # Добавляем в историю
            error_record = {
                "timestamp": datetime.now(),
                "error_type": error_type,
                "message": str(error),
                "context": context or {},
                "traceback": traceback.format_exc(),
            }
            self.error_history.append(error_record)

            # Ограничиваем размер истории
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size :]

            # Логируем ошибку
            logger.error(
                f"Error occurred: {error_type} - {str(error)}",
                extra={"context": context, "traceback": traceback.format_exc()},
            )

            # Вызываем зарегистрированный callback
            for error_class, callback in self.error_callbacks.items():
                if isinstance(error, error_class):
                    try:
                        callback(error, context)
                        return True
                    except Exception as callback_error:
                        logger.error(f"Error in error callback: {callback_error}")

            # Обработка по умолчанию
            return self._default_error_handler(error, context)

        except Exception as handler_error:
            logger.error(f"Error in error handler: {handler_error}")
            return False

    def _default_error_handler(
        self, error: Exception, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Обработчик ошибок по умолчанию."""

        if isinstance(error, TradingError):
            logger.error(
                "Trading error occurred",
                extra={"error": str(error), "context": context},
            )
            return True

        elif isinstance(error, ExchangeError):
            logger.error(
                "Exchange error occurred",
                extra={"error": str(error), "context": context},
            )
            return True

        elif isinstance(error, ConfigurationError):
            logger.error(
                "Configuration error occurred",
                extra={"error": str(error), "context": context},
            )
            return True

        else:
            logger.error(
                "Unexpected error occurred",
                extra={"error": str(error), "context": context},
            )
            return False

    def get_error_stats(self) -> Dict[str, Any]:
        """Получение статистики ошибок."""
        return {
            "counters": self.error_counters.copy(),
            "total_errors": sum(self.error_counters.values()),
            "recent_errors": self.error_history[-10:] if self.error_history else [],
        }

    def clear_history(self) -> None:
        """Очистка истории ошибок."""
        self.error_history.clear()
        self.error_counters.clear()


# Глобальный экземпляр обработчика ошибок
error_handler = ErrorHandler()


def handle_errors(func: Callable) -> Callable:
    """Декоратор для автоматической обработки ошибок."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
            }
            if error_handler.handle_error(e, context):
                raise Exception(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def async_handle_errors(func: Callable) -> Callable:
    """Асинхронный декоратор для автоматической обработки ошибок."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            context = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
            }
            if error_handler.handle_error(e, context):
                raise Exception(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    error_types: tuple = (ExchangeError,),
):
    """Декоратор для повторных попыток при ошибках."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_error

        return wrapper

    return decorator


def async_retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    error_types: tuple = (ExchangeError,),
):
    """Асинхронный декоратор для повторных попыток при ошибках."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_error

        return wrapper

    return decorator


# Регистрация обработчиков по умолчанию
def setup_default_error_handlers() -> None:
    """Настройка обработчиков ошибок по умолчанию."""

    def trading_error_handler(error: TradingError, context: Optional[Dict[str, Any]]):
        """Обработчик торговых ошибок."""
        logger.error(
            f"Trading error: {error.message}",
            extra={
                "error_code": error.error_code,
                "details": error.details,
                "context": context,
            },
        )

    def exchange_error_handler(error: ExchangeError, context: Optional[Dict[str, Any]]):
        """Обработчик ошибок биржи."""
        logger.error(
            f"Exchange error: {error.message}",
            extra={
                "error_code": error.error_code,
                "details": error.details,
                "context": context,
            },
        )

    def configuration_error_handler(
        error: ConfigurationError, context: Optional[Dict[str, Any]]
    ):
        """Обработчик ошибок конфигурации."""
        logger.error(
            f"Configuration error: {error.message}",
            extra={
                "error_code": error.error_code,
                "details": error.details,
                "context": context,
            },
        )

    # Регистрируем обработчики
    error_handler.register_error_callback(TradingError, trading_error_handler)
    error_handler.register_error_callback(ExchangeError, exchange_error_handler)
    error_handler.register_error_callback(
        ConfigurationError, configuration_error_handler
    )


# Инициализация при импорте
setup_default_error_handlers()
