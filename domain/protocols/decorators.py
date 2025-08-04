# -*- coding: utf-8 -*-
"""
Декораторы для протоколов и сервисов.
"""

import asyncio
import functools
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Типы для декораторов
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class ProtocolError(Exception):
    """Базовый класс для ошибок протоколов."""

    pass


class ProtocolTimeoutError(ProtocolError):
    """Ошибка таймаута протокола."""

    pass


class ProtocolCircuitBreakerError(ProtocolError):
    """Ошибка circuit breaker протокола."""

    pass


class ProtocolRateLimitError(ProtocolError):
    """Ошибка rate limit протокола."""

    pass


class ProtocolValidationError(ProtocolError):
    """Ошибка валидации протокола."""

    pass


class ProtocolCacheError(ProtocolError):
    """Ошибка кэша протокола."""

    pass


class ProtocolMetricsError(ProtocolError):
    """Ошибка метрик протокола."""

    pass


class RetryStrategy(Enum):
    """Стратегии повторных попыток."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


class CircuitState(Enum):
    """Состояния circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Конфигурация для retry декоратора."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_factor: float = 2.0
    exceptions: Set[Type[Exception]] = field(default_factory=lambda: {Exception})
    retry_on_result: Optional[Callable[[Any], bool]] = None


@dataclass
class TimeoutConfig:
    """Конфигурация для timeout декоратора."""

    timeout: float = 30.0
    default_result: Optional[Any] = None
    raise_on_timeout: bool = True


@dataclass
class CacheConfig:
    """Конфигурация для cache декоратора."""

    ttl: int = 300  # 5 минут
    max_size: int = 1000
    key_generator: Optional[Callable[..., str]] = None
    invalidate_on_exception: bool = True


@dataclass
class MetricsConfig:
    """Конфигурация для metrics декоратора."""

    enabled: bool = True
    track_time: bool = True
    track_memory: bool = False
    track_exceptions: bool = True
    custom_metrics: Dict[str, Callable[..., Any]] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Конфигурация для circuit breaker декоратора."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    monitor_interval: float = 10.0


@dataclass
class RateLimitConfig:
    """Конфигурация для rate limit декоратора."""

    max_calls: int = 100
    time_window: float = 60.0  # секунды
    burst_size: int = 10
    strategy: str = "token_bucket"  # "token_bucket", "leaky_bucket", "fixed_window"


class MetricsCollector:
    """Коллектор метрик для декораторов."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, List[float]] = {}
        self._errors: Dict[str, List[Exception]] = {}
        self._lock = threading.Lock()

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Увеличить счетчик."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def record_timer(self, name: str, duration: float) -> None:
        """Записать время выполнения."""
        with self._lock:
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(duration)

    def record_error(
        self, name: str, error: Exception, context: Dict[str, Any]
    ) -> None:
        """Записать ошибку."""
        with self._lock:
            if name not in self._errors:
                self._errors[name] = []
            self._errors[name].append(error)

    def get_metrics(self, name: str) -> Dict[str, Any]:
        """Получить метрики для функции."""
        with self._lock:
            counter = self._counters.get(name, 0)
            timers = self._timers.get(name, [])
            errors = self._errors.get(name, [])
            
            return {
                "calls": counter,
                "avg_time": sum(timers) / len(timers) if timers else 0,
                "min_time": min(timers) if timers else 0,
                "max_time": max(timers) if timers else 0,
                "errors": len(errors),
                "error_rate": len(errors) / counter if counter > 0 else 0,
            }


class CircuitBreaker:
    """Circuit breaker для защиты от каскадных сбоев."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Выполнить функцию с circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self.last_failure_time is not None and time.time() - self.last_failure_time > self.config.recovery_timeout:  # type: ignore[unreachable]
                    self.state = CircuitState.HALF_OPEN  # type: ignore[unreachable]
                else:
                    raise ProtocolCircuitBreakerError("Circuit breaker is OPEN")
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise e

    def _on_success(self) -> None:
        """Обработка успешного вызова."""
        self.failure_count = 0
        self.last_success_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Обработка неудачного вызова."""
        self.failure_count += 1
        self.last_failure_time = time.time()  # type: ignore[assignment]
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN


class RateLimiter:
    """Rate limiter для ограничения частоты вызовов."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.max_calls
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Получить разрешение на вызов."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            tokens_to_add = time_passed * (
                self.config.max_calls / self.config.time_window
            )
            self.tokens = min(self.config.max_calls, self.tokens + int(tokens_to_add))
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


# Глобальные экземпляры
_metrics_collector = MetricsCollector()
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_rate_limiters: Dict[str, RateLimiter] = {}


def retry(config: Optional[RetryConfig] = None) -> Callable[[F], F]:
    """
    Декоратор для повторных попыток с различными стратегиями.
    Args:
        config: Конфигурация retry механизма
    Returns:
        Декорированная функция
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(config.max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    # Проверяем результат если задан callback
                    if config.retry_on_result and config.retry_on_result(result):
                        raise ValueError("Result validation failed")
                    return result
                except tuple(config.exceptions) as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        break
                    # Вычисляем задержку
                    delay = _calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    # Проверяем результат если задан callback
                    if config.retry_on_result and config.retry_on_result(result):
                        raise ValueError("Result validation failed")
                    return result
                except tuple(config.exceptions) as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        break
                    # Вычисляем задержку
                    delay = _calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
            if last_exception:
                raise last_exception

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def timeout(config: Optional[TimeoutConfig] = None) -> Callable[[F], F]:
    """
    Декоратор для таймаута выполнения.
    Args:
        config: Конфигурация таймаута
    Returns:
        Декорированная функция
    """
    if config is None:
        config = TimeoutConfig()

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=config.timeout)
            except asyncio.TimeoutError:
                if config.raise_on_timeout:
                    raise ProtocolTimeoutError(f"Function {func.__name__} timed out after {config.timeout}s")
                return config.default_result

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Для синхронных функций используем threading
            result: List[Any] = [None]
            exception: List[Optional[Exception]] = [None]
            
            def target() -> None:
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=config.timeout)
            
            if thread.is_alive():
                if config.raise_on_timeout:
                    raise ProtocolTimeoutError(f"Function {func.__name__} timed out after {config.timeout}s")
                return config.default_result
            
            if exception[0]:
                raise exception[0]
            return result[0]

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def validate_input(validation_schema: Type[BaseModel]) -> Callable[[F], F]:
    """
    Декоратор для валидации входных данных.
    Args:
        validation_schema: Схема валидации Pydantic
    Returns:
        Декорированная функция
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Валидируем аргументы
                validated_args = validation_schema(*args, **kwargs)
                return await func(*validated_args.dict())
            except Exception as e:
                raise ProtocolValidationError("value", "", "format", f"Validation failed for {func.__name__}: {e}")

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Валидируем аргументы
                validated_args = validation_schema(*args, **kwargs)
                return func(*validated_args.dict())
            except Exception as e:
                raise ProtocolValidationError("value", "", "format", f"Validation failed for {func.__name__}: {e}")

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def cache(config: Optional[CacheConfig] = None) -> Callable[[F], F]:
    """
    Декоратор для кэширования результатов.
    Args:
        config: Конфигурация кэша
    Returns:
        Декорированная функция
    """
    if config is None:
        config = CacheConfig()
    
    cache_storage: Dict[str, Any] = {}
    cache_timestamps: Dict[str, float] = {}

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _generate_cache_key(func, args, kwargs, config)
            now = time.time()
            
            # Проверяем кэш
            if key in cache_storage and now - cache_timestamps[key] < config.ttl:
                return cache_storage[key]
            
            try:
                result = await func(*args, **kwargs)
                cache_storage[key] = result
                cache_timestamps[key] = now
                return result
            except Exception as e:
                if config.invalidate_on_exception and key in cache_storage:
                    del cache_storage[key]
                    del cache_timestamps[key]
                raise e

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _generate_cache_key(func, args, kwargs, config)
            now = time.time()
            
            # Проверяем кэш
            if key in cache_storage and now - cache_timestamps[key] < config.ttl:
                return cache_storage[key]
            
            try:
                result = func(*args, **kwargs)
                cache_storage[key] = result
                cache_timestamps[key] = now
                return result
            except Exception as e:
                if config.invalidate_on_exception and key in cache_storage:
                    del cache_storage[key]
                    del cache_timestamps[key]
                raise e

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def metrics(config: Optional[MetricsConfig] = None) -> Callable[[F], F]:
    """
    Декоратор для сбора метрик.
    Args:
        config: Конфигурация метрик
    Returns:
        Декорированная функция
    """
    if config is None:
        config = MetricsConfig()

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            start_memory = _get_memory_usage() if config.track_memory else 0
            
            try:
                result = await func(*args, **kwargs)
                
                if config.enabled:
                    duration = time.time() - start_time
                    _metrics_collector.record_timer(func.__name__, duration)
                    _metrics_collector.increment_counter(func.__name__)
                    
                    if config.track_memory:
                        end_memory = _get_memory_usage()
                        memory_usage = end_memory - start_memory
                        # Можно добавить метрики памяти
                
                return result
            except Exception as e:
                if config.enabled and config.track_exceptions:
                    _metrics_collector.record_error(func.__name__, e, {"args": args, "kwargs": kwargs})
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            start_memory = _get_memory_usage() if config.track_memory else 0
            
            try:
                result = func(*args, **kwargs)
                
                if config.enabled:
                    duration = time.time() - start_time
                    _metrics_collector.record_timer(func.__name__, duration)
                    _metrics_collector.increment_counter(func.__name__)
                    
                    if config.track_memory:
                        end_memory = _get_memory_usage()
                        memory_usage = end_memory - start_memory
                        # Можно добавить метрики памяти
                
                return result
            except Exception as e:
                if config.enabled and config.track_exceptions:
                    _metrics_collector.record_error(func.__name__, e, {"args": args, "kwargs": kwargs})
                raise

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def circuit_breaker(config: Optional[CircuitBreakerConfig] = None) -> Callable[[F], F]:
    """
    Декоратор для circuit breaker.
    Args:
        config: Конфигурация circuit breaker
    Returns:
        Декорированная функция
    """
    if config is None:
        config = CircuitBreakerConfig()

    def decorator(func: F) -> F:
        func_name = func.__name__
        if func_name not in _circuit_breakers:
            _circuit_breakers[func_name] = CircuitBreaker(config)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _circuit_breakers[func_name].call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Для синхронных функций создаем асинхронную обертку
            async def async_func(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            
            # Запускаем в event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_circuit_breakers[func_name].call(async_func, *args, **kwargs))
            finally:
                loop.close()

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def rate_limit(config: Optional[RateLimitConfig] = None) -> Callable[[F], F]:
    """
    Декоратор для rate limiting.
    Args:
        config: Конфигурация rate limit
    Returns:
        Декорированная функция
    """
    if config is None:
        config = RateLimitConfig()

    def decorator(func: F) -> F:
        func_name = func.__name__
        if func_name not in _rate_limiters:
            _rate_limiters[func_name] = RateLimiter(config)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not await _rate_limiters[func_name].acquire():
                raise ProtocolRateLimitError(f"Rate limit exceeded for {func_name}")
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Для синхронных функций создаем асинхронную обертку
            async def async_func(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            
            # Запускаем в event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if not loop.run_until_complete(_rate_limiters[func_name].acquire()):
                    raise ProtocolRateLimitError(f"Rate limit exceeded for {func_name}")
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


def log_operation(level: int = logging.INFO) -> Callable[[F], F]:
    """
    Декоратор для логирования операций.
    Args:
        level: Уровень логирования
    Returns:
        Декорированная функция
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.log(level, f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.log(level, f"Completed {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Failed {func.__name__}: {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.log(level, f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Completed {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Failed {func.__name__}: {e}")
                raise

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)

    return decorator


# Вспомогательные функции
def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Вычислить задержку для retry."""
    if config.strategy == RetryStrategy.FIXED:
        return config.base_delay
    elif config.strategy == RetryStrategy.LINEAR:
        return config.base_delay * (attempt + 1)
    elif config.strategy == RetryStrategy.EXPONENTIAL:
        delay = config.base_delay * (config.backoff_factor**attempt)
        return min(delay, config.max_delay)
    elif config.strategy == RetryStrategy.FIBONACCI:
        # Простая реализация Fibonacci
        fib = [1, 1]
        for i in range(attempt + 1):
            fib.append(fib[-1] + fib[-2])
        delay = config.base_delay * fib[attempt]
        return min(delay, config.max_delay)
    return config.base_delay  # type: ignore[unreachable]


def _generate_cache_key(
    func: Callable, args: tuple, kwargs: dict, config: CacheConfig
) -> str:
    """Генерировать ключ кэша."""
    if config.key_generator:
        return config.key_generator(*args, **kwargs)
    # Простая генерация ключа
    key_parts = [
        func.__module__,
        func.__name__,
        str(hash(args)),
        str(hash(frozenset(kwargs.items()))),
    ]
    return ":".join(key_parts)


def _get_memory_usage() -> int:
    """Получить использование памяти в байтах."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss  # type: ignore[no-any-return]
    except ImportError:
        return 0


# Функции для получения метрик
def get_metrics(func_name: str) -> Dict[str, Any]:
    """Получить метрики для функции."""
    return _metrics_collector.get_metrics(func_name)


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Получить все метрики."""
    return {
        name: _metrics_collector.get_metrics(name)
        for name in _metrics_collector._counters.keys()
    }


def reset_metrics() -> None:
    """Сбросить все метрики."""
    global _metrics_collector
    _metrics_collector = MetricsCollector()


def get_circuit_breaker_status(func_name: str) -> Dict[str, Any]:
    """Получить статус circuit breaker для функции."""
    if func_name not in _circuit_breakers:
        return {"status": "not_configured"}
    cb = _circuit_breakers[func_name]
    return {
        "state": cb.state.value,
        "failure_count": cb.failure_count,
        "last_failure_time": cb.last_failure_time,
        "last_success_time": cb.last_success_time,
    }


def get_rate_limiter_status(func_name: str) -> Dict[str, Any]:
    """Получить статус rate limiter для функции."""
    if func_name not in _rate_limiters:
        return {"status": "not_configured"}
    rl = _rate_limiters[func_name]
    return {
        "tokens": rl.tokens,
        "max_calls": rl.config.max_calls,
        "time_window": rl.config.time_window,
        "last_refill": rl.last_refill,
    }
