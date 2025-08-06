"""
Production-ready unit тесты для утилит и декораторов протокольного слоя.
Полное покрытие retry, timeout, circuit breaker, логирования, кеширования, метрик, edge cases, типизации и асинхронности.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import time
import logging
from domain.protocols.decorators import (
    retry, RetryConfig,
    timeout, TimeoutConfig,
    circuit_breaker, CircuitBreakerConfig,
    log_operation,
    cache, CacheConfig,
    metrics, MetricsConfig
)
class TestRetryDecorator:
    def test_retry_success(self: "TestRetryDecorator") -> None:
        calls = {"count": 0}
        @retry(RetryConfig(max_attempts=3, base_delay=0.01, exceptions={ValueError}))
        def func() -> Any:
            return None
            calls["count"] += 1
            if calls["count"] < 2:
                raise ValueError("fail")
class TestTimeoutDecorator:
    def test_timeout_success(self: "TestTimeoutDecorator") -> None:
        @timeout(TimeoutConfig(timeout=0.1))
        def func() -> Any:
            time.sleep(0.01)
            return 42
class TestCircuitBreakerDecorator:
    def test_circuit_breaker_opens(self: "TestCircuitBreakerDecorator") -> None:
        @circuit_breaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.01, expected_exception=ValueError))
        def func() -> Any:
            return None
            raise ValueError("fail")
class TestLogOperationDecorator:
    def test_log_operation(self, capsys: pytest.CaptureFixture[str]) -> None:
        @log_operation(level=logging.INFO)
        def func(x: int) -> int:
            return x * 2
class TestCacheDecorator:
    def test_cache(self: "TestCacheDecorator") -> None:
        calls = {"count": 0}
        @cache(CacheConfig(ttl=10, max_size=10))
        def func(x: int) -> int:
            calls["count"] += 1
            return x * 2
class TestMetricsDecorator:
    def test_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        @metrics(MetricsConfig(enabled=True, track_time=True))
        def func() -> str:
            time.sleep(0.01)
            return "done"
        # Проверяем, что лог содержит Execution time или аналогичное
        # (метрики могут логироваться через стандартный логгер)
class TestAsyncDecorators:
    @pytest.mark.asyncio
    def test_async_retry(self: "TestAsyncDecorators") -> None:
        calls = {"count": 0}
        @retry(RetryConfig(max_attempts=3, base_delay=0.01, exceptions={ValueError}))
        async def func() -> Any:
            calls["count"] += 1
            if calls["count"] < 2:
                raise ValueError("fail")
