"""
Circuit breaker implementation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    success_threshold: int = 2
    timeout: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit {self.name} moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")
            try:
                if asyncio.iscoroutinefunction(func):
                    if self.config.timeout:
                        result = await asyncio.wait_for(
                            func(*args, **kwargs), timeout=self.config.timeout
                        )
                    else:
                        result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    if self.config.timeout:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, func, *args, **kwargs),
                            timeout=self.config.timeout,
                        )
                    else:
                        result = await loop.run_in_executor(None, func, *args, **kwargs)
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure(e)
                raise

    async def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.success_count += 1
        self.last_success_time = time.time()
        if (
            self.state == CircuitState.HALF_OPEN
            and self.success_count >= self.config.success_threshold
        ):
            self.state = CircuitState.CLOSED
            self.success_count = 0
            logger.info(f"Circuit {self.name} moved to CLOSED")

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed execution."""
        if isinstance(exception, self.config.expected_exception):
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit {self.name} moved to OPEN after {self.failure_count} failures"
                )

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


async def get_breaker(
    name: str, config: CircuitBreakerConfig, fallback: Optional[Callable] = None
) -> CircuitBreaker:
    """Get or create circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()


def reset_breaker(name: str) -> bool:
    """Reset a specific circuit breaker."""
    if name in _circuit_breakers:
        breaker = _circuit_breakers[name]
        breaker.state = CircuitState.CLOSED
        breaker.failure_count = 0
        breaker.success_count = 0
        return True
    return False


def reset_all_breakers() -> None:
    """Reset all circuit breakers."""
    for breaker in _circuit_breakers.values():
        breaker.state = CircuitState.CLOSED
        breaker.failure_count = 0
        breaker.success_count = 0
