"""
Fallback strategies for circuit breaker.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FallbackError(BaseException):
    """Base exception for fallback operations."""

    pass


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute fallback strategy."""
        pass

    @abstractmethod
    def can_execute(self) -> bool:
        """Check if fallback can be executed."""
        pass


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""

    @abstractmethod
    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with retry strategy."""
        pass


class StaticFallback(FallbackStrategy):
    """Static fallback that returns a predefined value."""

    def __init__(self, value: Any):
        self.value = value
        self._execution_count = 0

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Return static value."""
        self._execution_count += 1
        logger.info(f"Static fallback executed {self._execution_count} times")
        return self.value

    def can_execute(self) -> bool:
        """Always can execute."""
        return True


class FunctionFallback(FallbackStrategy):
    """Fallback that calls a function."""

    def __init__(self, func: Callable[..., Any], max_executions: Optional[int] = None):
        self.func = func
        self.max_executions = max_executions
        self._execution_count = 0

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute fallback function."""
        if not self.can_execute():
            raise FallbackError("Fallback execution limit reached")

        self._execution_count += 1
        logger.info(f"Function fallback executed {self._execution_count} times")

        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.func, *args, **kwargs)

    def can_execute(self) -> bool:
        """Check if can execute based on limit."""
        if self.max_executions is None:
            return True
        return self._execution_count < self.max_executions


class CachedFallback(FallbackStrategy):
    """Fallback that returns cached value."""

    def __init__(self, cache_duration: float = 300.0):
        self.cache_duration = cache_duration
        self._cached_value = None
        self._cache_time = 0.0

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Return cached value if available and fresh."""
        current_time = time.time()

        if (
            self._cached_value is not None
            and current_time - self._cache_time < self.cache_duration
        ):
            logger.info("Returning cached fallback value")
            return self._cached_value

        logger.warning("No cached fallback value available")
        return None

    def can_execute(self) -> bool:
        """Check if cached value is available and fresh."""
        if self._cached_value is None:
            return False

        current_time = time.time()
        return current_time - self._cache_time < self.cache_duration

    def update_cache(self, value: Any) -> None:
        """Update cached value."""
        self._cached_value = value
        self._cache_time = time.time()
        logger.info("Fallback cache updated")


class DegradedFallback(FallbackStrategy):
    """Fallback that provides degraded functionality."""

    def __init__(self, degraded_func: Callable[..., Any], max_executions: Optional[int] = None):
        self.degraded_func = degraded_func
        self.max_executions = max_executions
        self._execution_count = 0

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute degraded function."""
        if not self.can_execute():
            raise FallbackError("Degraded fallback execution limit reached")

        self._execution_count += 1
        logger.warning(f"Degraded fallback executed {self._execution_count} times")

        if asyncio.iscoroutinefunction(self.degraded_func):
            return await self.degraded_func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.degraded_func, *args, **kwargs)

    def can_execute(self) -> bool:
        """Check if can execute based on limit."""
        if self.max_executions is None:
            return True
        return self._execution_count < self.max_executions


class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff retry strategy."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, *args, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    raise last_exception

        # if last_exception is not None:
        #     if isinstance(last_exception, BaseException):
        #         raise last_exception
        #     else:
        #         raise FallbackError(f"Unknown error: {last_exception}")
        # raise FallbackError("Unknown error in ExponentialBackoffRetry")


class FixedDelayRetry(RetryStrategy):
    """Fixed delay retry strategy."""

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with fixed delay retry."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, *args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {self.delay}s: {e}"
                    )
                    await asyncio.sleep(self.delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    if last_exception is not None:
                        if isinstance(last_exception, BaseException):
                            raise last_exception
                        else:
                            raise FallbackError(f"Unknown error: {last_exception}")
                    raise FallbackError("Unknown error in FixedDelayRetry")
        if last_exception is not None:
            if isinstance(last_exception, BaseException):
                raise last_exception
            else:
                raise FallbackError(f"Unknown error: {last_exception}")
        raise FallbackError("Unknown error in FixedDelayRetry")


class JitterRetry(RetryStrategy):
    """Retry strategy with jitter to avoid thundering herd."""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with jitter retry."""
        import random

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, *args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Calculate delay with jitter
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                    final_delay = delay + jitter
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {final_delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(final_delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    if last_exception is not None:
                        if isinstance(last_exception, BaseException):
                            raise last_exception
                        else:
                            raise FallbackError(f"Unknown error: {last_exception}")
                    raise FallbackError("Unknown error in JitterRetry")
        if last_exception is not None:
            if isinstance(last_exception, BaseException):
                raise last_exception
            else:
                raise FallbackError(f"Unknown error: {last_exception}")
        raise FallbackError("Unknown error in JitterRetry")


# Factory functions for common fallback patterns
def create_database_fallback(
    default_value: Any = None, use_cache: bool = True, cache_duration: float = 300.0
) -> FallbackStrategy:
    """Create fallback strategy for database operations."""
    if use_cache and default_value is not None:
        return CachedFallback(cache_duration)
    else:
        return StaticFallback(default_value)


def create_cache_fallback(
    degraded_func: Optional[Callable] = None, default_value: Any = None
) -> FallbackStrategy:
    """Create fallback strategy for cache operations."""
    if degraded_func:
        return DegradedFallback(degraded_func)
    else:
        return StaticFallback(default_value)


def create_api_fallback(
    degraded_func: Optional[Callable] = None, max_executions: Optional[int] = 10
) -> FallbackStrategy:
    """Create fallback strategy for API operations."""
    if degraded_func:
        return FunctionFallback(degraded_func, max_executions)
    else:
        return StaticFallback(None)


def create_trading_fallback(
    safe_mode_func: Optional[Callable] = None, max_executions: Optional[int] = 5
) -> FallbackStrategy:
    """Create fallback strategy for trading operations."""
    if safe_mode_func:
        return DegradedFallback(safe_mode_func, max_executions)
    else:
        return StaticFallback(None)


# Composite fallback strategies
class CompositeFallback(FallbackStrategy):
    """Fallback that tries multiple strategies in order."""

    def __init__(self, strategies: List[FallbackStrategy]):
        self.strategies = strategies
        self._current_strategy_index = 0

    async def execute(self, *args, **kwargs) -> Any:
        """Try strategies in order until one succeeds."""
        for i, strategy in enumerate(self.strategies):
            if strategy.can_execute():
                try:
                    result = await strategy.execute(*args, **kwargs)
                    self._current_strategy_index = i
                    return result
                except Exception as e:
                    logger.warning(f"Fallback strategy {i} failed: {e}")
                    continue

        raise FallbackError("All fallback strategies failed")

    def can_execute(self) -> bool:
        """Check if any strategy can execute."""
        return any(strategy.can_execute() for strategy in self.strategies)


class AdaptiveFallback(FallbackStrategy):
    """Adaptive fallback that learns from previous executions."""

    def __init__(self, strategies: List[FallbackStrategy]):
        self.strategies = strategies
        self._success_counts = [0] * len(strategies)
        self._failure_counts = [0] * len(strategies)
        self._current_strategy_index = 0

    async def execute(self, *args, **kwargs) -> Any:
        """Execute best available strategy."""
        # Find best strategy based on success rate
        best_strategy_index = self._get_best_strategy_index()

        if best_strategy_index is not None:
            strategy = self.strategies[best_strategy_index]
            if strategy.can_execute():
                try:
                    result = await strategy.execute(*args, **kwargs)
                    self._success_counts[best_strategy_index] += 1
                    self._current_strategy_index = best_strategy_index
                    return result
                except Exception as e:
                    self._failure_counts[best_strategy_index] += 1
                    logger.warning(f"Best fallback strategy failed: {e}")

        # Try any available strategy
        for i, strategy in enumerate(self.strategies):
            if strategy.can_execute():
                try:
                    result = await strategy.execute(*args, **kwargs)
                    self._success_counts[i] += 1
                    self._current_strategy_index = i
                    return result
                except Exception as e:
                    self._failure_counts[i] += 1
                    logger.warning(f"Fallback strategy {i} failed: {e}")

        raise FallbackError("All fallback strategies failed")

    def can_execute(self) -> bool:
        """Check if any strategy can execute."""
        return any(strategy.can_execute() for strategy in self.strategies)

    def _get_best_strategy_index(self) -> Optional[int]:
        """Get index of strategy with best success rate."""
        best_index = None
        best_rate = -1.0

        for i in range(len(self.strategies)):
            total = self._success_counts[i] + self._failure_counts[i]
            if total > 0:
                success_rate = self._success_counts[i] / total
                if success_rate > best_rate and self.strategies[i].can_execute():
                    best_rate = success_rate
                    best_index = i

        return best_index

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all strategies."""
        stats = {}
        for i, strategy in enumerate(self.strategies):
            total = self._success_counts[i] + self._failure_counts[i]
            success_rate = self._success_counts[i] / total if total > 0 else 0.0

            stats[f"strategy_{i}"] = {
                "success_count": self._success_counts[i],
                "failure_count": self._failure_counts[i],
                "success_rate": success_rate,
                "can_execute": strategy.can_execute(),
            }

        return stats
