"""
Утилиты для промышленных протоколов домена.
Обеспечивают вспомогательные функции для работы с протоколами.
"""

import asyncio
import functools
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Awaitable, Tuple, Type
from uuid import UUID

from domain.exceptions.protocol_exceptions import (
    ConfigurationError,
    ProtocolError,
    RetryExhaustedError,
    TimeoutError,
)
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
    ValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# DECORATORS
# ============================================================================


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (ProtocolError,),
) -> Callable[[F], F]:
    """
    Декоратор для повторных попыток при ошибках.

    Args:
        max_retries: Максимальное количество попыток
        delay: Начальная задержка в секундах
        backoff_factor: Множитель задержки
        exceptions: Кортеж исключений для перехвата
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor

            if last_exception is not None:
                raise last_exception
            else:
                raise ProtocolError("Retry exhausted without specific exception")

        return wrapper

    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Декоратор для установки таймаута.

    Args:
        seconds: Таймаут в секундах
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Operation {func.__name__} timed out after {seconds} seconds",
                    timeout_seconds=seconds,
                    operation=func.__name__,
                )

        return wrapper

    return decorator


def validate_entity_id(entity_id: Union[UUID, str]) -> UUID:
    """
    Валидация ID сущности.

    Args:
        entity_id: ID сущности

    Returns:
        UUID: Валидный UUID

    Raises:
        ValidationError: Неверный формат ID
    """
    if isinstance(entity_id, str):
        try:
            return UUID(entity_id)
        except ValueError:
            raise ValidationError(f"Invalid entity ID format: {entity_id}")
    elif isinstance(entity_id, UUID):
        return entity_id
    else:
        raise ValidationError(f"Entity ID must be UUID or string, got {type(entity_id)}")


def validate_symbol(symbol: Union[str, Symbol]) -> Symbol:
    """
    Валидация торгового символа.

    Args:
        symbol: Торговый символ

    Returns:
        Symbol: Валидный символ

    Raises:
        ValidationError: Неверный формат символа
    """
    if isinstance(symbol, str):
        if not symbol or len(symbol) > 20:
            raise ValidationError(f"Invalid symbol format: {symbol}")
        return Symbol(symbol)
    else:
        raise ValidationError(f"Symbol must be string or Symbol, got {type(symbol)}")


# ============================================================================
# CACHE UTILITIES
# ============================================================================


class ProtocolCache:
    """Кэш для протоколов."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires_at"]:
                return entry["value"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Установка значения в кэш."""
        ttl = ttl_seconds or self.ttl_seconds
        self.cache[key] = {
            "value": value,
            "expires_at": datetime.now() + timedelta(seconds=ttl),
        }

    def delete(self, key: str) -> None:
        """Удаление значения из кэша."""
        if key in self.cache:
            del self.cache[key]

    def clear(self) -> None:
        """Очистка кэша."""
        self.cache.clear()

    def cleanup_expired(self) -> None:
        """Очистка истекших записей."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items() if now >= entry["expires_at"]
        ]
        for key in expired_keys:
            del self.cache[key]


# ============================================================================
# METRICS UTILITIES
# ============================================================================


class ProtocolMetrics:
    """Метрики для протоколов."""

    def __init__(self) -> None:
        self.operation_counts: Dict[str, int] = {}
        self.operation_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_operation: Dict[str, datetime] = {}

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """Запись метрики операции."""
        # Счетчики операций
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1

        # Время выполнения
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)

        # Ошибки
        if not success and error_type:
            error_key = f"{operation}_{error_type}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Последняя операция
        self.last_operation[operation] = datetime.now()

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Получение статистики операции."""
        times = self.operation_times.get(operation, [])
        return {
            "count": self.operation_counts.get(operation, 0),
            "avg_time": sum(times) / len(times) if times else 0,
            "min_time": min(times) if times else 0,
            "max_time": max(times) if times else 0,
            "last_operation": self.last_operation.get(operation),
            "error_count": sum(
                count
                for key, count in self.error_counts.items()
                if key.startswith(operation)
            ),
        }

    def reset(self) -> None:
        """Сброс метрик."""
        self.operation_counts.clear()
        self.operation_times.clear()
        self.error_counts.clear()
        self.last_operation.clear()


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Валидация конфигурации.

    Args:
        config: Конфигурация
        required_keys: Обязательные ключи

    Raises:
        ConfigurationError: Неверная конфигурация
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration keys: {missing_keys}",
            config_key="required_keys",
            config_value=missing_keys,
        )


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Объединение конфигураций.

    Args:
        base_config: Базовая конфигурация
        override_config: Конфигурация для переопределения

    Returns:
        Dict[str, Any]: Объединенная конфигурация
    """
    result = base_config.copy()
    result.update(override_config)
    return result


# ============================================================================
# ASYNC UTILITIES
# ============================================================================


async def with_timeout(
    coro: Awaitable[Any], timeout_seconds: float, operation_name: str
) -> Any:
    """
    Выполнение корутины с таймаутом.

    Args:
        coro: Корутина для выполнения
        timeout_seconds: Таймаут в секундах
        operation_name: Название операции

    Returns:
        Any: Результат выполнения

    Raises:
        TimeoutError: Превышен таймаут
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Operation {operation_name} timed out after {timeout_seconds} seconds",
            timeout_seconds=timeout_seconds,
            operation=operation_name,
        )


async def batch_operation(
    items: List[Any], operation: Callable[[Any], Awaitable[Any]], batch_size: int = 10, max_concurrent: int = 5
) -> List[Any]:
    """
    Пакетная обработка элементов.

    Args:
        items: Список элементов
        operation: Операция для выполнения
        batch_size: Размер пакета
        max_concurrent: Максимальное количество одновременных операций

    Returns:
        List[Any]: Результаты обработки
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def process_item(item: Any) -> Any:
        async with semaphore:
            return await operation(item)

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch], return_exceptions=True
        )
        results.extend(batch_results)

    return results


# ============================================================================
# LOGGING UTILITIES
# ============================================================================


def log_operation(
    operation: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[Union[UUID, str]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Логирование операции протокола.

    Args:
        operation: Название операции
        entity_type: Тип сущности
        entity_id: ID сущности
        extra_data: Дополнительные данные
    """
    log_data = {"operation": operation, "timestamp": datetime.now().isoformat()}

    if entity_type:
        log_data["entity_type"] = entity_type
    if entity_id:
        log_data["entity_id"] = str(entity_id)
    if extra_data:
        log_data.update(extra_data)

    logger.info(f"Protocol operation: {log_data}")


def log_error(
    error: Exception,
    operation: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[Union[UUID, str]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Логирование ошибки протокола.

    Args:
        error: Ошибка
        operation: Название операции
        entity_type: Тип сущности
        entity_id: ID сущности
        extra_data: Дополнительные данные
    """
    log_data = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
    }

    if entity_type:
        log_data["entity_type"] = entity_type
    if entity_id:
        log_data["entity_id"] = str(entity_id)
    if extra_data:
        log_data.update(extra_data)

    logger.error(f"Protocol error: {log_data}", exc_info=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Decorators
    "retry_on_error",
    "timeout",
    # Validation
    "validate_entity_id",
    "validate_symbol",
    # Cache
    "ProtocolCache",
    # Metrics
    "ProtocolMetrics",
    # Configuration
    "validate_config",
    "merge_configs",
    # Async utilities
    "with_timeout",
    "batch_operation",
    # Logging
    "log_operation",
    "log_error",
]
