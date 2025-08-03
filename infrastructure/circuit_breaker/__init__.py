"""
Circuit breaker module.
"""

from .breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    get_all_breakers,
    get_breaker,
    reset_all_breakers,
    reset_breaker,
)
from .decorators import (
    async_circuit_breaker,
    cache_circuit_breaker,
    circuit_breaker,
    database_circuit_breaker,
    external_api_circuit_breaker,
    retry_with_circuit_breaker,
    sync_circuit_breaker,
    trading_circuit_breaker,
    with_fallback,
)
from .fallback import (
    AdaptiveFallback,
    CachedFallback,
    CompositeFallback,
    DegradedFallback,
    ExponentialBackoffRetry,
    FallbackStrategy,
    FixedDelayRetry,
    FunctionFallback,
    JitterRetry,
    RetryStrategy,
    StaticFallback,
)

__all__ = [
    # Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "get_breaker",
    "get_all_breakers",
    "reset_breaker",
    "reset_all_breakers",
    # Fallback
    "FallbackStrategy",
    "RetryStrategy",
    "StaticFallback",
    "FunctionFallback",
    "CachedFallback",
    "DegradedFallback",
    "ExponentialBackoffRetry",
    "FixedDelayRetry",
    "JitterRetry",
    "CompositeFallback",
    "AdaptiveFallback",
    # Decorators
    "circuit_breaker",
    "async_circuit_breaker",
    "sync_circuit_breaker",
    "database_circuit_breaker",
    "cache_circuit_breaker",
    "external_api_circuit_breaker",
    "trading_circuit_breaker",
    "with_fallback",
    "retry_with_circuit_breaker",
]
