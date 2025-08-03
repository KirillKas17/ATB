"""
Decorators for circuit breaker functionality.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

from .breaker import CircuitBreakerConfig, get_breaker

logger = logging.getLogger(__name__)


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 2,
    timeout: Optional[float] = None,
    fallback: Optional[Callable] = None,
) -> Callable[[Callable], Callable]:
    """Universal circuit breaker decorator that works with both sync and async functions."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold,
                timeout=timeout,
            )
            breaker = await get_breaker(circuit_name, config, fallback)
            return await breaker.call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            async def async_func() -> Any:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception,
                    success_threshold=success_threshold,
                    timeout=timeout,
                )
                breaker = await get_breaker(circuit_name, config, fallback)
                return await breaker.call(func, *args, **kwargs)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def async_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 2,
    timeout: Optional[float] = None,
    fallback: Optional[Callable] = None,
) -> Callable[[Callable], Callable]:
    """Decorator specifically for async functions."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold,
                timeout=timeout,
            )
            breaker = await get_breaker(circuit_name, config, fallback)
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def sync_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 2,
    timeout: Optional[float] = None,
    fallback: Optional[Callable] = None,
) -> Callable[[Callable], Callable]:
    """Decorator specifically for sync functions."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold,
                timeout=timeout,
            )

            async def async_func() -> Any:
                breaker = await get_breaker(circuit_name, config, fallback)
                return await breaker.call(func, *args, **kwargs)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        return wrapper

    return decorator


# Specialized decorators for common use cases
def database_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
    timeout: Optional[float] = 10.0,
) -> Callable[[Callable], Callable]:
    """Circuit breaker optimized for database operations."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout,
            )
            breaker = await get_breaker(circuit_name, config)
            return await breaker.call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            async def async_func() -> Any:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    timeout=timeout,
                )
                breaker = await get_breaker(circuit_name, config)
                return await breaker.call(func, *args, **kwargs)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 15.0,
    timeout: Optional[float] = 5.0,
) -> Callable[[Callable], Callable]:
    """Circuit breaker optimized for cache operations."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout,
            )
            breaker = await get_breaker(circuit_name, config)
            return await breaker.call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            async def async_func() -> Any:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    timeout=timeout,
                )
                breaker = await get_breaker(circuit_name, config)
                return await breaker.call(func, *args, **kwargs)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def external_api_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 3,
    recovery_timeout: float = 60.0,
    timeout: Optional[float] = 30.0,
) -> Callable[[Callable], Callable]:
    """Circuit breaker optimized for external API calls."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout,
            )
            breaker = await get_breaker(circuit_name, config)
            return await breaker.call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            async def async_func() -> Any:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    timeout=timeout,
                )
                breaker = await get_breaker(circuit_name, config)
                return await breaker.call(func, *args, **kwargs)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def trading_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 2,
    recovery_timeout: float = 120.0,
    timeout: Optional[float] = 60.0,
) -> Callable[[Callable], Callable]:
    """Circuit breaker optimized for trading operations."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout,
            )
            breaker = await get_breaker(circuit_name, config)
            return await breaker.call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            async def async_func() -> Any:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    timeout=timeout,
                )
                breaker = await get_breaker(circuit_name, config)
                return await breaker.call(func, *args, **kwargs)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Fallback decorators
def with_fallback(fallback_func: Callable) -> Callable[[Callable], Callable]:
    """Decorator that provides a fallback function when the main function fails."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Retry decorator that works with circuit breaker
def retry_with_circuit_breaker(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
) -> Callable[[Callable], Callable]:
    """Decorator that combines retry logic with circuit breaker pattern."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = circuit_breaker_config or CircuitBreakerConfig()
            breaker = await get_breaker(f"{func.__module__}.{func.__name__}", config)

            for attempt in range(max_retries):
                try:
                    return await breaker.call(func, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = retry_delay * (backoff_factor ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = circuit_breaker_config or CircuitBreakerConfig()

            async def async_func() -> Any:
                breaker = await get_breaker(f"{func.__module__}.{func.__name__}", config)

                for attempt in range(max_retries):
                    try:
                        return await breaker.call(func, *args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        delay = retry_delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)

            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, run coroutine directly
                    return asyncio.run_coroutine_threadsafe(async_func(), loop).result()
                else:
                    return asyncio.run(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
