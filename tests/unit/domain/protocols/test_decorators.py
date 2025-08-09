"""
Unit тесты для domain/protocols/decorators.py.

Покрывает:
- Конфигурации (RetryConfig, TimeoutConfig, CacheConfig, MetricsConfig, CircuitBreakerConfig, RateLimitConfig)
- MetricsCollector
- CircuitBreaker
- RateLimiter
- Декораторы (retry, timeout, validate_input, cache, metrics, circuit_breaker, rate_limit, log_operation)
- Утилиты
- Обработку ошибок
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel

from domain.protocols.decorators import (
    ProtocolError,
    ProtocolTimeoutError,
    ProtocolCircuitBreakerError,
    ProtocolRateLimitError,
    ProtocolValidationError,
    ProtocolCacheError,
    ProtocolMetricsError,
    RetryStrategy,
    CircuitState,
    RetryConfig,
    TimeoutConfig,
    CacheConfig,
    MetricsConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    MetricsCollector,
    CircuitBreaker,
    RateLimiter,
    retry,
    timeout,
    validate_input,
    cache,
    metrics,
    circuit_breaker,
    rate_limit,
    log_operation,
    _calculate_delay,
    _generate_cache_key,
    _get_memory_usage,
    get_metrics,
    get_all_metrics,
    reset_metrics,
    get_circuit_breaker_status,
    get_rate_limiter_status,
)


class TestRetryStrategy:
    """Тесты для RetryStrategy."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert RetryStrategy.FIXED.value == "fixed"
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.FIBONACCI.value == "fibonacci"


class TestCircuitState:
    """Тесты для CircuitState."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestRetryConfig:
    """Тесты для RetryConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.max_delay == 60.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.backoff_factor == 2.0
        assert Exception in config.exceptions

    def test_custom_values(self):
        """Тест пользовательских значений."""
        config = RetryConfig(max_attempts=5, max_delay=30.0, strategy=RetryStrategy.LINEAR, backoff_factor=1.5)
        assert config.max_attempts == 5
        assert config.max_delay == 30.0
        assert config.strategy == RetryStrategy.LINEAR
        assert config.backoff_factor == 1.5


class TestTimeoutConfig:
    """Тесты для TimeoutConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = TimeoutConfig()
        assert config.timeout == 30.0
        assert config.default_result is None
        assert config.raise_on_timeout is True

    def test_custom_values(self):
        """Тест пользовательских значений."""
        config = TimeoutConfig(timeout=10.0, default_result="default", raise_on_timeout=False)
        assert config.timeout == 10.0
        assert config.default_result == "default"
        assert config.raise_on_timeout is False


class TestCacheConfig:
    """Тесты для CacheConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = CacheConfig()
        assert config.ttl == 300
        assert config.max_size == 1000
        assert config.key_generator is None
        assert config.invalidate_on_exception is True

    def test_custom_values(self):
        """Тест пользовательских значений."""

        def custom_key_generator(*args, **kwargs):
            return "custom_key"

        config = CacheConfig(ttl=600, max_size=500, key_generator=custom_key_generator, invalidate_on_exception=False)
        assert config.ttl == 600
        assert config.max_size == 500
        assert config.key_generator == custom_key_generator
        assert config.invalidate_on_exception is False


class TestMetricsConfig:
    """Тесты для MetricsConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.track_time is True
        assert config.track_memory is False
        assert config.track_exceptions is True
        assert config.custom_metrics == {}

    def test_custom_values(self):
        """Тест пользовательских значений."""
        custom_metrics = {"custom": lambda: 42}
        config = MetricsConfig(
            enabled=False, track_time=False, track_memory=True, track_exceptions=False, custom_metrics=custom_metrics
        )
        assert config.enabled is False
        assert config.track_time is False
        assert config.track_memory is True
        assert config.track_exceptions is False
        assert config.custom_metrics == custom_metrics


class TestCircuitBreakerConfig:
    """Тесты для CircuitBreakerConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.expected_exception == Exception
        assert config.monitor_interval == 10.0

    def test_custom_values(self):
        """Тест пользовательских значений."""
        config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=30.0, expected_exception=ValueError, monitor_interval=5.0
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.expected_exception == ValueError
        assert config.monitor_interval == 5.0


class TestRateLimitConfig:
    """Тесты для RateLimitConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = RateLimitConfig()
        assert config.max_calls == 100
        assert config.time_window == 60.0
        assert config.burst_size == 10
        assert config.strategy == "token_bucket"

    def test_custom_values(self):
        """Тест пользовательских значений."""
        config = RateLimitConfig(max_calls=50, time_window=30.0, burst_size=5, strategy="leaky_bucket")
        assert config.max_calls == 50
        assert config.time_window == 30.0
        assert config.burst_size == 5
        assert config.strategy == "leaky_bucket"


class TestMetricsCollector:
    """Тесты для MetricsCollector."""

    @pytest.fixture
    def metrics_collector(self):
        """Фикстура сборщика метрик."""
        return MetricsCollector()

    def test_initialization(self, metrics_collector):
        """Тест инициализации."""
        assert metrics_collector._metrics == {}

    def test_increment_counter(self, metrics_collector):
        """Тест увеличения счетчика."""
        metrics_collector.increment_counter("test_counter", 5)
        metrics_collector.increment_counter("test_counter", 3)

        metrics = metrics_collector.get_metrics("test_counter")
        assert metrics["count"] == 8

    def test_record_timer(self, metrics_collector):
        """Тест записи таймера."""
        metrics_collector.record_timer("test_timer", 0.1)
        metrics_collector.record_timer("test_timer", 0.2)

        metrics = metrics_collector.get_metrics("test_timer")
        assert metrics["count"] == 2
        assert metrics["total_time"] == 0.3
        assert metrics["avg_time"] == 0.15

    def test_record_error(self, metrics_collector):
        """Тест записи ошибки."""
        error = ValueError("Test error")
        context = {"operation": "test"}

        metrics_collector.record_error("test_error", error, context)

        metrics = metrics_collector.get_metrics("test_error")
        assert metrics["error_count"] == 1
        assert "ValueError" in metrics["error_types"]

    def test_get_metrics_nonexistent(self, metrics_collector):
        """Тест получения метрик несуществующего счетчика."""
        metrics = metrics_collector.get_metrics("nonexistent")
        assert metrics["count"] == 0
        assert metrics["total_time"] == 0.0
        assert metrics["error_count"] == 0


class TestCircuitBreaker:
    """Тесты для CircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Фикстура circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        return CircuitBreaker(config)

    def test_initialization(self, circuit_breaker):
        """Тест инициализации."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    async def test_successful_call(self, circuit_breaker):
        """Тест успешного вызова."""

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    async def test_failure_threshold_reached(self, circuit_breaker):
        """Тест достижения порога ошибок."""

        async def failing_func():
            raise ValueError("Test error")

        # Первая ошибка
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)

        # Вторая ошибка - circuit breaker открывается
        with pytest.raises(ProtocolCircuitBreakerError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    async def test_recovery_timeout(self, circuit_breaker):
        """Тест таймаута восстановления."""

        async def failing_func():
            raise ValueError("Test error")

        # Достигаем порога ошибок
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)
        with pytest.raises(ProtocolCircuitBreakerError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Ждем восстановления
        await asyncio.sleep(1.1)

        # Теперь должно быть в состоянии HALF_OPEN
        assert circuit_breaker.state == CircuitState.HALF_OPEN


class TestRateLimiter:
    """Тесты для RateLimiter."""

    @pytest.fixture
    def rate_limiter(self):
        """Фикстура rate limiter."""
        config = RateLimitConfig(max_calls=5, time_window=1.0, burst_size=2)
        return RateLimiter(config)

    def test_initialization(self, rate_limiter):
        """Тест инициализации."""
        assert rate_limiter.config.max_calls == 5
        assert rate_limiter.config.time_window == 1.0

    async def test_acquire_within_limit(self, rate_limiter):
        """Тест получения разрешения в пределах лимита."""
        for i in range(5):
            result = await rate_limiter.acquire()
            assert result is True

    async def test_acquire_exceeds_limit(self, rate_limiter):
        """Тест превышения лимита."""
        # Используем все разрешения
        for i in range(5):
            await rate_limiter.acquire()

        # Следующее должно быть отклонено
        result = await rate_limiter.acquire()
        assert result is False

    async def test_reset_after_time_window(self, rate_limiter):
        """Тест сброса после временного окна."""
        # Используем все разрешения
        for i in range(5):
            await rate_limiter.acquire()

        # Ждем истечения временного окна
        await asyncio.sleep(1.1)

        # Теперь должно снова работать
        result = await rate_limiter.acquire()
        assert result is True


class TestRetryDecorator:
    """Тесты для декоратора retry."""

    @retry(RetryConfig(max_attempts=3))
    async def test_function_success(self) -> str:
        """Тестовая функция, которая завершается успешно."""
        return "success"

    @retry(RetryConfig(max_attempts=2))
    async def test_function_failure(self) -> str:
        """Тестовая функция, которая всегда падает."""
        raise ValueError("Test error")

    @retry(RetryConfig(max_attempts=3))
    async def test_function_success_after_retry(self) -> str:
        """Тестовая функция, которая успешна после нескольких попыток."""
        if not hasattr(self, "_attempts"):
            self._attempts = 0
        self._attempts += 1

        if self._attempts < 3:
            raise ValueError(f"Attempt {self._attempts}")
        return "success"

    async def test_successful_execution(self: "TestRetryDecorator") -> None:
        """Тест успешного выполнения."""
        result = await self.test_function_success()
        assert result == "success"

    async def test_failure_after_max_retries(self: "TestRetryDecorator") -> None:
        """Тест неудачи после максимального количества попыток."""
        with pytest.raises(ValueError):
            await self.test_function_failure()

    async def test_success_after_retries(self: "TestRetryDecorator") -> None:
        """Тест успеха после нескольких попыток."""
        result = await self.test_function_success_after_retry()
        assert result == "success"


class TestTimeoutDecorator:
    """Тесты для декоратора timeout."""

    @timeout(TimeoutConfig(timeout=0.5))
    async def test_function_fast(self) -> str:
        """Тестовая функция, которая выполняется быстро."""
        await asyncio.sleep(0.1)
        return "success"

    @timeout(TimeoutConfig(timeout=0.1))
    async def test_function_slow(self) -> str:
        """Тестовая функция, которая выполняется медленно."""
        await asyncio.sleep(0.5)
        return "success"

    @timeout(TimeoutConfig(timeout=0.1, default_result="default", raise_on_timeout=False))
    async def test_function_slow_with_default(self) -> str:
        """Тестовая функция с результатом по умолчанию."""
        await asyncio.sleep(0.5)
        return "success"

    async def test_successful_execution(self: "TestTimeoutDecorator") -> None:
        """Тест успешного выполнения."""
        result = await self.test_function_fast()
        assert result == "success"

    async def test_timeout_exception(self: "TestTimeoutDecorator") -> None:
        """Тест исключения таймаута."""
        with pytest.raises(ProtocolTimeoutError):
            await self.test_function_slow()

    async def test_timeout_with_default_result(self: "TestTimeoutDecorator") -> None:
        """Тест таймаута с результатом по умолчанию."""
        result = await self.test_function_slow_with_default()
        assert result == "default"


class TestValidateInputDecorator:
    """Тесты для декоратора validate_input."""

    class TestSchema(BaseModel):
        name: str
        age: int

    @validate_input(TestSchema)
    async def test_function(self, data: TestSchema) -> str:
        """Тестовая функция с валидацией входных данных."""
        return f"Hello {data.name}"

    async def test_valid_input(self: "TestValidateInputDecorator") -> None:
        """Тест валидного ввода."""
        data = self.TestSchema(name="John", age=30)
        result = await self.test_function(data)
        assert result == "Hello John"

    async def test_invalid_input(self: "TestValidateInputDecorator") -> None:
        """Тест невалидного ввода."""
        with pytest.raises(ProtocolValidationError):
            await self.test_function({"name": "John", "age": "invalid"})


class TestCacheDecorator:
    """Тесты для декоратора cache."""

    @cache(CacheConfig(ttl=300))
    async def test_function(self, param: str) -> str:
        """Тестовая функция с кэшированием."""
        return f"result for {param}"

    async def test_cache_hit(self: "TestCacheDecorator") -> None:
        """Тест попадания в кэш."""
        result1 = await self.test_function("test")
        result2 = await self.test_function("test")
        assert result1 == result2

    async def test_cache_miss(self: "TestCacheDecorator") -> None:
        """Тест промаха кэша."""
        result1 = await self.test_function("test1")
        result2 = await self.test_function("test2")
        assert result1 != result2


class TestMetricsDecorator:
    """Тесты для декоратора metrics."""

    @metrics(MetricsConfig(enabled=True, track_time=True))
    async def test_function(self) -> str:
        """Тестовая функция с метриками."""
        await asyncio.sleep(0.1)
        return "success"

    async def test_metrics_recording(self: "TestMetricsDecorator") -> None:
        """Тест записи метрик."""
        result = await self.test_function()
        assert result == "success"

        # Проверяем, что метрики записаны
        metrics_data = get_metrics("test_function")
        assert metrics_data["count"] > 0
        assert metrics_data["total_time"] > 0


class TestCircuitBreakerDecorator:
    """Тесты для декоратора circuit_breaker."""

    @circuit_breaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0))
    async def test_function_success(self) -> str:
        """Тестовая функция, которая завершается успешно."""
        return "success"

    @circuit_breaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0))
    async def test_function_failure(self) -> str:
        """Тестовая функция, которая всегда падает."""
        raise ValueError("Test error")

    async def test_successful_execution(self: "TestCircuitBreakerDecorator") -> None:
        """Тест успешного выполнения."""
        result = await self.test_function_success()
        assert result == "success"

    async def test_circuit_breaker_opens(self: "TestCircuitBreakerDecorator") -> None:
        """Тест открытия circuit breaker."""
        # Первые два вызова должны вызвать ошибку
        with pytest.raises(ValueError):
            await self.test_function_failure()
        with pytest.raises(ValueError):
            await self.test_function_failure()

        # Третий вызов должен вызвать ошибку circuit breaker
        with pytest.raises(ProtocolCircuitBreakerError):
            await self.test_function_failure()


class TestRateLimitDecorator:
    """Тесты для декоратора rate_limit."""

    @rate_limit(RateLimitConfig(max_calls=3, time_window=1.0))
    async def test_function(self) -> str:
        """Тестовая функция с ограничением скорости."""
        return "success"

    async def test_rate_limiting(self: "TestRateLimitDecorator") -> None:
        """Тест ограничения скорости."""
        # Первые три вызова должны быть успешными
        for i in range(3):
            result = await self.test_function()
            assert result == "success"

        # Четвертый вызов должен быть отклонен
        with pytest.raises(ProtocolRateLimitError):
            await self.test_function()


class TestLogOperationDecorator:
    """Тесты для декоратора log_operation."""

    @log_operation()
    async def test_function(self) -> str:
        """Тестовая функция с логированием."""
        return "success"

    async def test_logging(self: "TestLogOperationDecorator") -> None:
        """Тест логирования."""
        with patch("domain.protocols.decorators.logger") as mock_logger:
            result = await self.test_function()
            assert result == "success"
            mock_logger.info.assert_called()


class TestUtilityFunctions:
    """Тесты для утилитарных функций."""

    def test_calculate_delay_fixed(self):
        """Тест расчета задержки для фиксированной стратегии."""
        config = RetryConfig(strategy=RetryStrategy.FIXED, backoff_factor=1.0)
        delay = _calculate_delay(1, config)
        assert delay == 1.0

    def test_calculate_delay_exponential(self):
        """Тест расчета задержки для экспоненциальной стратегии."""
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, backoff_factor=2.0)
        delay = _calculate_delay(2, config)
        assert delay == 4.0

    def test_generate_cache_key(self):
        """Тест генерации ключа кэша."""

        def test_func(a, b):
            pass

        config = CacheConfig()
        key = _generate_cache_key(test_func, (1, 2), {"c": 3}, config)
        assert isinstance(key, str)
        assert "test_func" in key

    def test_get_memory_usage(self):
        """Тест получения использования памяти."""
        usage = _get_memory_usage()
        assert isinstance(usage, int)
        assert usage >= 0

    def test_get_metrics(self):
        """Тест получения метрик."""
        metrics = get_metrics("test_function")
        assert isinstance(metrics, dict)

    def test_get_all_metrics(self):
        """Тест получения всех метрик."""
        all_metrics = get_all_metrics()
        assert isinstance(all_metrics, dict)

    def test_reset_metrics(self):
        """Тест сброса метрик."""
        reset_metrics()
        all_metrics = get_all_metrics()
        assert all_metrics == {}

    def test_get_circuit_breaker_status(self):
        """Тест получения статуса circuit breaker."""
        status = get_circuit_breaker_status("test_function")
        assert isinstance(status, dict)

    def test_get_rate_limiter_status(self):
        """Тест получения статуса rate limiter."""
        status = get_rate_limiter_status("test_function")
        assert isinstance(status, dict)


class TestDecoratorsIntegration:
    """Интеграционные тесты декораторов."""

    @retry(RetryConfig(max_attempts=2))
    @timeout(TimeoutConfig(timeout=0.5))
    @cache(CacheConfig(ttl=300))
    @metrics(MetricsConfig(enabled=True))
    @log_operation()
    async def test_function_integration(self, param: str) -> str:
        """Тестовая функция с множественными декораторами."""
        await asyncio.sleep(0.1)
        return f"result for {param}"

    async def test_multiple_decorators(self: "TestDecoratorsIntegration") -> None:
        """Тест множественных декораторов."""
        result = await self.test_function_integration("test")
        assert result == "result for test"

        # Проверяем кэширование
        result2 = await self.test_function_integration("test")
        assert result2 == result


class TestDecoratorsErrorHandling:
    """Тесты обработки ошибок декораторов."""

    @retry(RetryConfig(max_attempts=1))
    async def test_retry_with_custom_exception(self: "TestDecoratorsErrorHandling") -> None:
        """Тест retry с пользовательским исключением."""
        raise ProtocolError("Custom error")

    async def test_retry_custom_exception(self: "TestDecoratorsErrorHandling") -> None:
        """Тест retry с пользовательским исключением."""
        with pytest.raises(ProtocolError):
            await self.test_retry_with_custom_exception()

    @timeout(TimeoutConfig(timeout=0.1))
    async def test_timeout_with_slow_operation(self: "TestDecoratorsErrorHandling") -> None:
        """Тест таймаута с медленной операцией."""
        await asyncio.sleep(0.5)
        return "success"

    async def test_timeout_slow_operation(self: "TestDecoratorsErrorHandling") -> None:
        """Тест таймаута с медленной операцией."""
        with pytest.raises(ProtocolTimeoutError):
            await self.test_timeout_with_slow_operation()

    @circuit_breaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1.0))
    async def test_circuit_breaker_with_error(self: "TestDecoratorsErrorHandling") -> None:
        """Тест circuit breaker с ошибкой."""
        raise ValueError("Test error")

    async def test_circuit_breaker_error(self: "TestDecoratorsErrorHandling") -> None:
        """Тест circuit breaker с ошибкой."""
        with pytest.raises(ValueError):
            await self.test_circuit_breaker_with_error()

        with pytest.raises(ProtocolCircuitBreakerError):
            await self.test_circuit_breaker_with_error()
