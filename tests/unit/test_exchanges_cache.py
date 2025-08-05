"""
Unit тесты для ExchangeCache
"""
import asyncio
import time
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.external_services.exchanges.cache import ExchangeCache
class TestExchangeCache:
    """Тесты для ExchangeCache."""
    @pytest.fixture
    def cache(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Экземпляр ExchangeCache."""
        return ExchangeCache(max_size=10, ttl=1)
    @pytest.fixture
    def small_cache(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Маленький кэш для тестов."""
        return ExchangeCache(max_size=3, ttl=0.1)
    def test_init(self: "TestExchangeCache") -> None:
        """Тест инициализации."""
        cache = ExchangeCache(max_size=100, ttl=60)
        assert cache.max_size == 100
        assert cache.ttl == 60
        assert cache.cache == {}
        assert cache.lock is not None
    def test_init_default_values(self: "TestExchangeCache") -> None:
        """Тест инициализации с значениями по умолчанию."""
        cache = ExchangeCache()
        assert cache.max_size == 1000
        assert cache.ttl == 60
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache) -> None:
        """Тест установки и получения значения."""
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache) -> None:
        """Тест получения несуществующего ключа."""
        value = await cache.get("nonexistent_key")
        assert value is None
    @pytest.mark.asyncio
    async def test_get_expired_value(self, cache) -> None:
        """Тест получения истекшего значения."""
        await cache.set("test_key", "test_value")
        # Ждем, пока значение истечет
        await asyncio.sleep(1.1)
        value = await cache.get("test_key")
        assert value is None
    @pytest.mark.asyncio
    async def test_set_multiple_values(self, cache) -> None:
        """Тест установки нескольких значений."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
    @pytest.mark.asyncio
    async def test_set_different_types(self, cache) -> None:
        """Тест установки значений разных типов."""
        await cache.set("string", "hello")
        await cache.set("number", 42)
        await cache.set("list", [1, 2, 3])
        await cache.set("dict", {"key": "value"})
        await cache.set("none", None)
        assert await cache.get("string") == "hello"
        assert await cache.get("number") == 42
        assert await cache.get("list") == [1, 2, 3]
        assert await cache.get("dict") == {"key": "value"}
        assert await cache.get("none") is None
    @pytest.mark.asyncio
    async def test_max_size_eviction(self, small_cache) -> None:
        """Тест вытеснения при достижении максимального размера."""
        # Заполняем кэш до максимума
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")
        assert await small_cache.get_size() == 3
        # Добавляем еще один элемент
        await small_cache.set("key4", "value4")
        # Самый старый элемент должен быть удален
        assert await small_cache.get_size() == 3
        assert await small_cache.get("key1") is None
        assert await small_cache.get("key2") == "value2"
        assert await small_cache.get("key3") == "value3"
        assert await small_cache.get("key4") == "value4"
    @pytest.mark.asyncio
    async def test_max_size_eviction_with_expired(self, small_cache) -> None:
        """Тест вытеснения с учетом истекших значений."""
        # Устанавливаем значения с разными TTL
        await small_cache.set("key1", "value1")
        await asyncio.sleep(0.05)
        await small_cache.set("key2", "value2")
        await asyncio.sleep(0.05)
        await small_cache.set("key3", "value3")
        # Ждем, пока первое значение истечет
        await asyncio.sleep(0.1)
        # Добавляем новый элемент
        await small_cache.set("key4", "value4")
        # Проверяем, что истекшее значение удалено
        assert await small_cache.get("key1") is None
        assert await small_cache.get("key2") == "value2"
        assert await small_cache.get("key3") == "value3"
        assert await small_cache.get("key4") == "value4"
    @pytest.mark.asyncio
    async def test_clear(self, cache) -> None:
        """Тест очистки кэша."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        assert await cache.get_size() == 2
        await cache.clear()
        assert await cache.get_size() == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    @pytest.mark.asyncio
    async def test_remove_existing_key(self, cache) -> None:
        """Тест удаления существующего ключа."""
        await cache.set("test_key", "test_value")
        assert await cache.get_size() == 1
        result = await cache.remove("test_key")
        assert result is True
        assert await cache.get_size() == 0
        assert await cache.get("test_key") is None
    @pytest.mark.asyncio
    async def test_remove_nonexistent_key(self, cache) -> None:
        """Тест удаления несуществующего ключа."""
        result = await cache.remove("nonexistent_key")
        assert result is False
    @pytest.mark.asyncio
    async def test_get_size(self, cache) -> None:
        """Тест получения размера кэша."""
        assert await cache.get_size() == 0
        await cache.set("key1", "value1")
        assert await cache.get_size() == 1
        await cache.set("key2", "value2")
        assert await cache.get_size() == 2
        await cache.remove("key1")
        assert await cache.get_size() == 1
    @pytest.mark.asyncio
    async def test_get_keys(self, cache) -> None:
        """Тест получения всех ключей."""
        assert await cache.get_keys() == []
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        keys = await cache.get_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
    @pytest.mark.asyncio
    async def test_get_keys_with_expired(self, cache) -> None:
        """Тест получения ключей с истекшими значениями."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        # Ждем, пока первое значение истечет
        await asyncio.sleep(1.1)
        keys = await cache.get_keys()
        # Истекшие ключи должны быть удалены
        assert len(keys) == 1
        assert "key2" in keys
        assert "key1" not in keys
    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache) -> None:
        """Тест конкурентного доступа."""
        async def set_value(key, value) -> Any:
            await cache.set(key, value)
            return await cache.get(key)
        async def get_value(key) -> Any:
            return await cache.get(key)
        # Выполняем конкурентные операции
        tasks = [
            set_value(f"key{i}", f"value{i}")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(result == f"value{i}" for i, result in enumerate(results))
        assert await cache.get_size() == 5
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, cache) -> None:
        """Тест конкурентного чтения и записи."""
        async def writer() -> Any:
            for i in range(10):
                await cache.set(f"key{i}", f"value{i}")
                await asyncio.sleep(0.01)
        async def reader() -> Any:
            for i in range(10):
                await cache.get(f"key{i}")
                await asyncio.sleep(0.01)
        # Запускаем читателя и писателя одновременно
        await asyncio.gather(writer(), reader())
        # Проверяем, что операции завершились без ошибок
        assert await cache.get_size() > 0
    @pytest.mark.asyncio
    async def test_cache_performance(self, cache) -> None:
        """Тест производительности кэша."""
        # Устанавливаем много значений
        start_time = time.time()
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}")
        set_time = time.time() - start_time
        # Читаем значения
        start_time = time.time()
        for i in range(100):
            await cache.get(f"key{i}")
        get_time = time.time() - start_time
        # Операции должны выполняться быстро
        assert set_time < 1.0
        assert get_time < 1.0
    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self, cache) -> None:
        """Тест эффективности использования памяти."""
        # Устанавливаем значения до достижения лимита
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")
        assert await cache.get_size() == 10
        # Добавляем еще один элемент
        await cache.set("overflow", "overflow_value")
        # Размер должен остаться в пределах лимита
        assert await cache.get_size() == 10
        # Самый старый элемент должен быть удален
        assert await cache.get("key0") is None
        assert await cache.get("overflow") == "overflow_value"
    @pytest.mark.asyncio
    async def test_cache_ttl_accuracy(self, small_cache) -> None:
        """Тест точности TTL."""
        await small_cache.set("test_key", "test_value")
        # Проверяем, что значение доступно сразу
        assert await small_cache.get("test_key") == "test_value"
        # Ждем половину TTL
        await asyncio.sleep(0.05)
        # Значение все еще должно быть доступно
        assert await small_cache.get("test_key") == "test_value"
        # Ждем полный TTL
        await asyncio.sleep(0.1)
        # Значение должно истечь
        assert await small_cache.get("test_key") is None
    @pytest.mark.asyncio
    def test_cache_edge_cases(self: "TestExchangeCache") -> None:
        """Тест граничных случаев."""
        # Кэш с нулевым размером
        zero_cache = ExchangeCache(max_size=0, ttl=1)
        await zero_cache.set("key", "value")
        assert await zero_cache.get_size() == 0
        assert await zero_cache.get("key") is None
        # Кэш с очень большим TTL
        long_ttl_cache = ExchangeCache(max_size=10, ttl=3600)  # 1 час
        await long_ttl_cache.set("key", "value")
        assert await long_ttl_cache.get("key") == "value"
        # Кэш с очень коротким TTL
        short_ttl_cache = ExchangeCache(max_size=10, ttl=0.001)  # 1 мс
        await short_ttl_cache.set("key", "value")
        await asyncio.sleep(0.01)  # Ждем больше TTL
        assert await short_ttl_cache.get("key") is None
    @pytest.mark.asyncio
    async def test_cache_thread_safety(self, cache) -> None:
        """Тест потокобезопасности."""
        async def mixed_operations() -> Any:
            await cache.set("key", "value")
            value = await cache.get("key")
            size = await cache.get_size()
            keys = await cache.get_keys()
            removed = await cache.remove("key")
            return value, size, len(keys), removed
        # Выполняем смешанные операции конкурентно
        tasks = [mixed_operations() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        # Все операции должны завершиться без ошибок
        assert len(results) == 10
        assert all(isinstance(result, tuple) for result in results) 
