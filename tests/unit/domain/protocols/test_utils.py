"""
Unit тесты для domain/protocols/utils.py.

Покрывает:
- Декораторы (retry_on_error, timeout)
- Валидаторы (validate_entity_id, validate_symbol)
- ProtocolCache
- ProtocolMetrics
- Утилиты конфигурации
- Утилиты операций
- Утилиты логирования
- Обработку ошибок
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4

from domain.protocols.utils import (
    retry_on_error,
    timeout,
    validate_entity_id,
    validate_symbol,
    ProtocolCache,
    ProtocolMetrics,
    validate_config,
    merge_configs,
    with_timeout,
    batch_operation,
    log_operation,
    log_error,
)
from domain.exceptions.protocol_exceptions import (
    ConfigurationError,
    ProtocolError,
    RetryExhaustedError,
    TimeoutError,
)
from domain.type_definitions import (
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


class TestRetryOnErrorDecorator:
    """Тесты для декоратора retry_on_error."""

    @retry_on_error(max_retries=3, delay=0.1)
    async def test_function_success(self) -> str:
        """Тестовая функция, которая завершается успешно."""
        return "success"

    @retry_on_error(max_retries=2, delay=0.1)
    async def test_function_failure(self) -> str:
        """Тестовая функция, которая всегда падает."""
        raise ProtocolError("Test error")

    @retry_on_error(max_retries=3, delay=0.1)
    async def test_function_success_after_retry(self) -> str:
        """Тестовая функция, которая успешна после нескольких попыток."""
        if not hasattr(self, '_attempts'):
            self._attempts = 0
        self._attempts += 1
        
        if self._attempts < 3:
            raise ProtocolError(f"Attempt {self._attempts}")
        return "success"

    async def test_successful_execution(self):
        """Тест успешного выполнения."""
        result = await self.test_function_success()
        assert result == "success"

    async def test_failure_after_max_retries(self):
        """Тест неудачи после максимального количества попыток."""
        with pytest.raises(ProtocolError):
            await self.test_function_failure()

    async def test_success_after_retries(self):
        """Тест успеха после нескольких попыток."""
        result = await self.test_function_success_after_retry()
        assert result == "success"

    async def test_custom_exceptions(self):
        """Тест с пользовательскими исключениями."""
        @retry_on_error(max_retries=2, delay=0.1, exceptions=(ValueError,))
        async def test_custom_error():
            raise ValueError("Custom error")

        with pytest.raises(ValueError):
            await test_custom_error()


class TestTimeoutDecorator:
    """Тесты для декоратора timeout."""

    @timeout(0.5)
    async def test_function_fast(self) -> str:
        """Тестовая функция, которая выполняется быстро."""
        await asyncio.sleep(0.1)
        return "success"

    @timeout(0.1)
    async def test_function_slow(self) -> str:
        """Тестовая функция, которая выполняется медленно."""
        await asyncio.sleep(0.5)
        return "success"

    async def test_successful_execution(self):
        """Тест успешного выполнения."""
        result = await self.test_function_fast()
        assert result == "success"

    async def test_timeout_exception(self):
        """Тест исключения таймаута."""
        with pytest.raises(TimeoutError):
            await self.test_function_slow()


class TestValidateEntityId:
    """Тесты для validate_entity_id."""

    def test_valid_uuid_string(self):
        """Тест валидной строки UUID."""
        uuid_str = str(uuid4())
        result = validate_entity_id(uuid_str)
        assert isinstance(result, uuid4().__class__)

    def test_valid_uuid_object(self):
        """Тест валидного объекта UUID."""
        uuid_obj = uuid4()
        result = validate_entity_id(uuid_obj)
        assert result == uuid_obj

    def test_invalid_uuid_string(self):
        """Тест невалидной строки UUID."""
        with pytest.raises(ValidationError):
            validate_entity_id("invalid-uuid")

    def test_empty_string(self):
        """Тест пустой строки."""
        with pytest.raises(ValidationError):
            validate_entity_id("")

    def test_none_value(self):
        """Тест None значения."""
        with pytest.raises(ValidationError):
            validate_entity_id(None)


class TestValidateSymbol:
    """Тесты для validate_symbol."""

    def test_valid_symbol_string(self):
        """Тест валидной строки символа."""
        symbol_str = "BTC/USD"
        result = validate_symbol(symbol_str)
        assert isinstance(result, Symbol)
        assert result == "BTC/USD"

    def test_valid_symbol_object(self):
        """Тест валидного объекта Symbol."""
        symbol_obj = Symbol("ETH/USD")
        result = validate_symbol(symbol_obj)
        assert result == symbol_obj

    def test_invalid_symbol_format(self):
        """Тест невалидного формата символа."""
        with pytest.raises(ValidationError):
            validate_symbol("INVALID")

    def test_empty_string(self):
        """Тест пустой строки."""
        with pytest.raises(ValidationError):
            validate_symbol("")

    def test_none_value(self):
        """Тест None значения."""
        with pytest.raises(ValidationError):
            validate_symbol(None)


class TestProtocolCache:
    """Тесты для ProtocolCache."""

    @pytest.fixture
    def cache(self):
        """Фикстура кэша."""
        return ProtocolCache(ttl_seconds=300)

    def test_initialization(self, cache):
        """Тест инициализации."""
        assert cache.ttl_seconds == 300
        assert cache._cache == {}

    def test_set_and_get(self, cache):
        """Тест установки и получения значения."""
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"

    def test_get_nonexistent_key(self, cache):
        """Тест получения несуществующего ключа."""
        result = cache.get("nonexistent")
        assert result is None

    def test_set_with_custom_ttl(self, cache):
        """Тест установки с пользовательским TTL."""
        cache.set("test_key", "test_value", ttl_seconds=60)
        result = cache.get("test_key")
        assert result == "test_value"

    def test_delete(self, cache):
        """Тест удаления ключа."""
        cache.set("test_key", "test_value")
        cache.delete("test_key")
        result = cache.get("test_key")
        assert result is None

    def test_clear(self, cache):
        """Тест очистки кэша."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cleanup_expired(self, cache):
        """Тест очистки истекших записей."""
        # Устанавливаем короткий TTL
        cache.ttl_seconds = 0.1
        cache.set("test_key", "test_value")
        
        # Ждем истечения TTL
        import time
        time.sleep(0.2)
        
        cache.cleanup_expired()
        result = cache.get("test_key")
        assert result is None

    def test_expired_entry(self, cache):
        """Тест истекшей записи."""
        # Устанавливаем короткий TTL
        cache.ttl_seconds = 0.1
        cache.set("test_key", "test_value")
        
        # Ждем истечения TTL
        import time
        time.sleep(0.2)
        
        result = cache.get("test_key")
        assert result is None


class TestProtocolMetrics:
    """Тесты для ProtocolMetrics."""

    @pytest.fixture
    def metrics(self):
        """Фикстура метрик."""
        return ProtocolMetrics()

    def test_initialization(self, metrics):
        """Тест инициализации."""
        assert metrics._operations == {}

    def test_record_operation_success(self, metrics):
        """Тест записи успешной операции."""
        metrics.record_operation("test_op", 0.1, success=True)
        
        stats = metrics.get_operation_stats("test_op")
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0
        assert stats["avg_duration"] == 0.1

    def test_record_operation_failure(self, metrics):
        """Тест записи неудачной операции."""
        metrics.record_operation("test_op", 0.2, success=False, error_type="TimeoutError")
        
        stats = metrics.get_operation_stats("test_op")
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 1
        assert stats["error_types"]["TimeoutError"] == 1

    def test_multiple_operations(self, metrics):
        """Тест множественных операций."""
        metrics.record_operation("test_op", 0.1, success=True)
        metrics.record_operation("test_op", 0.2, success=True)
        metrics.record_operation("test_op", 0.3, success=False, error_type="ValueError")
        
        stats = metrics.get_operation_stats("test_op")
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["avg_duration"] == 0.2

    def test_get_nonexistent_operation_stats(self, metrics):
        """Тест получения статистики несуществующей операции."""
        stats = metrics.get_operation_stats("nonexistent")
        assert stats["total_calls"] == 0
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 0

    def test_reset(self, metrics):
        """Тест сброса метрик."""
        metrics.record_operation("test_op", 0.1, success=True)
        metrics.reset()
        
        stats = metrics.get_operation_stats("test_op")
        assert stats["total_calls"] == 0


class TestValidateConfig:
    """Тесты для validate_config."""

    def test_valid_config(self):
        """Тест валидной конфигурации."""
        config = {"key1": "value1", "key2": "value2", "key3": "value3"}
        required_keys = ["key1", "key2"]
        
        # Не должно вызывать исключение
        validate_config(config, required_keys)

    def test_missing_required_key(self):
        """Тест отсутствующего обязательного ключа."""
        config = {"key1": "value1"}
        required_keys = ["key1", "key2"]
        
        with pytest.raises(ConfigurationError):
            validate_config(config, required_keys)

    def test_empty_config(self):
        """Тест пустой конфигурации."""
        config = {}
        required_keys = ["key1"]
        
        with pytest.raises(ConfigurationError):
            validate_config(config, required_keys)

    def test_none_config(self):
        """Тест None конфигурации."""
        required_keys = ["key1"]
        
        with pytest.raises(ConfigurationError):
            validate_config(None, required_keys)


class TestMergeConfigs:
    """Тесты для merge_configs."""

    def test_merge_configs(self):
        """Тест слияния конфигураций."""
        base_config = {"key1": "value1", "key2": "value2"}
        override_config = {"key2": "new_value2", "key3": "value3"}
        
        result = merge_configs(base_config, override_config)
        
        assert result["key1"] == "value1"
        assert result["key2"] == "new_value2"
        assert result["key3"] == "value3"

    def test_merge_with_empty_override(self):
        """Тест слияния с пустой переопределяющей конфигурацией."""
        base_config = {"key1": "value1", "key2": "value2"}
        override_config = {}
        
        result = merge_configs(base_config, override_config)
        
        assert result == base_config

    def test_merge_with_empty_base(self):
        """Тест слияния с пустой базовой конфигурацией."""
        base_config = {}
        override_config = {"key1": "value1", "key2": "value2"}
        
        result = merge_configs(base_config, override_config)
        
        assert result == override_config

    def test_deep_merge(self):
        """Тест глубокого слияния."""
        base_config = {"nested": {"key1": "value1", "key2": "value2"}}
        override_config = {"nested": {"key2": "new_value2", "key3": "value3"}}
        
        result = merge_configs(base_config, override_config)
        
        assert result["nested"]["key1"] == "value1"
        assert result["nested"]["key2"] == "new_value2"
        assert result["nested"]["key3"] == "value3"


class TestWithTimeout:
    """Тесты для with_timeout."""

    async def test_successful_operation(self):
        """Тест успешной операции."""
        async def test_coro():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await with_timeout(test_coro(), 1.0, "test_operation")
        assert result == "success"

    async def test_timeout_exception(self):
        """Тест исключения таймаута."""
        async def slow_coro():
            await asyncio.sleep(2.0)
            return "success"
        
        with pytest.raises(TimeoutError):
            await with_timeout(slow_coro(), 0.5, "test_operation")

    async def test_operation_exception(self):
        """Тест исключения операции."""
        async def failing_coro():
            raise ValueError("Operation failed")
        
        with pytest.raises(ValueError):
            await with_timeout(failing_coro(), 1.0, "test_operation")


class TestBatchOperation:
    """Тесты для batch_operation."""

    async def test_batch_operation_success(self):
        """Тест успешной пакетной операции."""
        items = [1, 2, 3, 4, 5]
        
        async def process_item(item):
            await asyncio.sleep(0.01)
            return item * 2
        
        results = await batch_operation(items, process_item, batch_size=2, max_concurrent=3)
        assert results == [2, 4, 6, 8, 10]

    async def test_batch_operation_with_failures(self):
        """Тест пакетной операции с ошибками."""
        items = [1, 2, 3, 4, 5]
        
        async def process_item(item):
            if item == 3:
                raise ValueError("Item 3 failed")
            await asyncio.sleep(0.01)
            return item * 2
        
        with pytest.raises(ValueError):
            await batch_operation(items, process_item, batch_size=2, max_concurrent=3)

    async def test_empty_items(self):
        """Тест с пустым списком элементов."""
        items = []
        
        async def process_item(item):
            return item * 2
        
        results = await batch_operation(items, process_item)
        assert results == []

    async def test_single_item(self):
        """Тест с одним элементом."""
        items = [5]
        
        async def process_item(item):
            await asyncio.sleep(0.01)
            return item * 2
        
        results = await batch_operation(items, process_item)
        assert results == [10]


class TestLogOperation:
    """Тесты для log_operation."""

    @patch('domain.protocols.utils.logger')
    def test_log_operation_basic(self, mock_logger):
        """Тест базового логирования операции."""
        log_operation("test_operation")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_operation" in call_args

    @patch('domain.protocols.utils.logger')
    def test_log_operation_with_entity(self, mock_logger):
        """Тест логирования операции с сущностью."""
        entity_id = uuid4()
        log_operation("test_operation", entity_type="Order", entity_id=entity_id)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_operation" in call_args
        assert "Order" in call_args
        assert str(entity_id) in call_args

    @patch('domain.protocols.utils.logger')
    def test_log_operation_with_extra_data(self, mock_logger):
        """Тест логирования операции с дополнительными данными."""
        extra_data = {"key1": "value1", "key2": "value2"}
        log_operation("test_operation", extra_data=extra_data)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_operation" in call_args
        assert "key1" in call_args
        assert "value1" in call_args


class TestLogError:
    """Тесты для log_error."""

    @patch('domain.protocols.utils.logger')
    def test_log_error_basic(self, mock_logger):
        """Тест базового логирования ошибки."""
        error = ValueError("Test error")
        log_error(error, "test_operation")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "test_operation" in call_args
        assert "Test error" in call_args

    @patch('domain.protocols.utils.logger')
    def test_log_error_with_entity(self, mock_logger):
        """Тест логирования ошибки с сущностью."""
        error = ValueError("Test error")
        entity_id = uuid4()
        log_error(error, "test_operation", entity_type="Order", entity_id=entity_id)
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "test_operation" in call_args
        assert "Order" in call_args
        assert str(entity_id) in call_args

    @patch('domain.protocols.utils.logger')
    def test_log_error_with_extra_data(self, mock_logger):
        """Тест логирования ошибки с дополнительными данными."""
        error = ValueError("Test error")
        extra_data = {"key1": "value1", "key2": "value2"}
        log_error(error, "test_operation", extra_data=extra_data)
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "test_operation" in call_args
        assert "key1" in call_args
        assert "value1" in call_args


class TestUtilsIntegration:
    """Интеграционные тесты утилит."""

    async def test_retry_with_timeout(self):
        """Тест комбинации retry и timeout."""
        @retry_on_error(max_retries=2, delay=0.1)
        @timeout(0.5)
        async def test_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_function()
        assert result == "success"

    async def test_cache_with_metrics(self):
        """Тест комбинации кэша и метрик."""
        cache = ProtocolCache(ttl_seconds=300)
        metrics = ProtocolMetrics()
        
        # Записываем операцию в метрики
        metrics.record_operation("cache_get", 0.01, success=True)
        
        # Используем кэш
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        
        assert result == "test_value"
        
        # Проверяем метрики
        stats = metrics.get_operation_stats("cache_get")
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1

    async def test_batch_operation_with_timeout(self):
        """Тест пакетной операции с таймаутом."""
        items = [1, 2, 3, 4, 5]
        
        async def process_item(item):
            await asyncio.sleep(0.01)
            return item * 2
        
        results = await with_timeout(
            batch_operation(items, process_item, batch_size=2, max_concurrent=3),
            1.0,
            "batch_operation"
        )
        
        assert results == [2, 4, 6, 8, 10]


class TestUtilsErrorHandling:
    """Тесты обработки ошибок утилит."""

    async def test_retry_with_custom_exception(self):
        """Тест retry с пользовательским исключением."""
        @retry_on_error(max_retries=2, delay=0.1, exceptions=(ValueError,))
        async def test_function():
            raise ValueError("Custom error")
        
        with pytest.raises(ValueError):
            await test_function()

    async def test_timeout_with_fast_operation(self):
        """Тест таймаута с быстрой операцией."""
        @timeout(1.0)
        async def test_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_function()
        assert result == "success"

    def test_validate_config_with_invalid_input(self):
        """Тест валидации конфигурации с невалидным вводом."""
        with pytest.raises(ConfigurationError):
            validate_config("invalid", ["key1"])

    def test_merge_configs_with_invalid_input(self):
        """Тест слияния конфигураций с невалидным вводом."""
        with pytest.raises(TypeError):
            merge_configs("invalid", {"key1": "value1"})

    async def test_batch_operation_with_invalid_input(self):
        """Тест пакетной операции с невалидным вводом."""
        with pytest.raises(TypeError):
            await batch_operation("invalid", lambda x: x) 