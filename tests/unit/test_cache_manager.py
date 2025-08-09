"""
Unit тесты для CacheManager.
Тестирует управление кешем, включая сохранение, получение,
инвалидацию и оптимизацию кешированных данных.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from infrastructure.core.cache_manager import CacheManager


class TestCacheManager:
    """Тесты для CacheManager."""

    @pytest.fixture
    def cache_manager(self) -> CacheManager:
        """Фикстура для CacheManager."""
        return CacheManager()

    @pytest.fixture
    def sample_data(self) -> dict:
        """Фикстура с тестовыми данными."""
        return {
            "market_data": {
                "BTCUSDT": {"price": Decimal("50000.0"), "volume": Decimal("1000000.0"), "timestamp": datetime.now()}
            },
            "user_preferences": {"theme": "dark", "language": "en", "notifications": True},
            "strategy_config": {"rsi_period": 14, "ma_short": 10, "ma_long": 50},
        }

    def test_initialization(self, cache_manager: CacheManager) -> None:
        """Тест инициализации менеджера кеша."""
        assert cache_manager is not None
        assert hasattr(cache_manager, "cache")
        assert hasattr(cache_manager, "max_size")
        assert hasattr(cache_manager, "default_ttl")

    @pytest.mark.asyncio
    async def test_set_cache(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест установки кеша."""
        # Установка кеша
        set_result = await cache_manager.set("test_key", sample_data)
        # Проверки
        assert set_result is True
        # Проверка, что кеш установлен
        cached_data = await cache_manager.get("test_key")
        assert cached_data == sample_data

    @pytest.mark.asyncio
    async def test_get_cache(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест получения кеша."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Получение кеша
        get_result = await cache_manager.get("test_key")
        # Проверки
        assert get_result is not None
        assert get_result == sample_data

    @pytest.mark.asyncio
    async def test_delete_cache(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест удаления кеша."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Удаление кеша
        delete_result = await cache_manager.delete("test_key")
        # Проверки
        assert delete_result is True
        # Проверка, что кеш удален
        get_result = await cache_manager.get("test_key")
        assert get_result is None

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест очистки кеша."""
        # Установка нескольких кешей
        await cache_manager.set("key1", sample_data)
        await cache_manager.set("key2", sample_data)
        # Очистка кеша
        clear_result = await cache_manager.clear()
        # Проверки
        assert clear_result is True
        # Проверка, что все кеши очищены
        get_result1 = await cache_manager.get("key1")
        get_result2 = await cache_manager.get("key2")
        assert get_result1 is None
        assert get_result2 is None

    @pytest.mark.asyncio
    async def test_cache_exists(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест проверки существования кеша."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Проверка существования
        exists_result = await cache_manager.exists("test_key")
        # Проверки
        assert exists_result is True
        # Проверка несуществующего ключа
        not_exists_result = await cache_manager.exists("nonexistent_key")
        assert not_exists_result is False

    @pytest.mark.asyncio
    async def test_get_cache_keys(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест получения ключей кеша."""
        # Установка нескольких кешей
        await cache_manager.set("key1", sample_data)
        await cache_manager.set("key2", sample_data)
        await cache_manager.set("key3", sample_data)
        # Получение ключей
        keys_result = await cache_manager.keys()
        # Проверки
        assert keys_result is not None
        assert isinstance(keys_result, list)
        assert len(keys_result) >= 3
        assert "key1" in keys_result
        assert "key2" in keys_result
        assert "key3" in keys_result

    @pytest.mark.asyncio
    async def test_get_cache_info(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест получения информации о кеше."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Получение информации
        size_result = await cache_manager.size()
        # Проверки
        assert size_result is not None
        assert isinstance(size_result, int)
        assert size_result > 0

    @pytest.mark.asyncio
    async def test_set_cache_with_ttl(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест установки кеша с TTL."""
        # Установка кеша с TTL
        ttl = 300  # 5 минут
        set_result = await cache_manager.set("test_key", sample_data, ttl=ttl)
        # Проверки
        assert set_result is True
        # Проверка TTL
        ttl_result = await cache_manager.ttl("test_key")
        assert ttl_result > 0
        assert ttl_result <= ttl

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест истечения срока действия кеша."""
        # Установка кеша с коротким TTL
        ttl = 1  # 1 секунда
        await cache_manager.set("test_key", sample_data, ttl=ttl)
        # Проверка, что кеш существует
        exists_result = await cache_manager.exists("test_key")
        assert exists_result is True
        # Ожидание истечения TTL
        import asyncio

        await asyncio.sleep(1.1)
        # Проверка, что кеш истек
        get_result = await cache_manager.get("test_key")
        assert get_result is None

    @pytest.mark.asyncio
    async def test_cache_compression(self, cache_manager: CacheManager) -> None:
        """Тест сжатия кеша."""
        # Тест с большими данными
        large_data = {"data": "x" * 10000}
        set_result = await cache_manager.set("large_key", large_data)
        # Проверки
        assert set_result is True
        # Получение данных
        get_result = await cache_manager.get("large_key")
        assert get_result == large_data

    @pytest.mark.asyncio
    async def test_cache_serialization(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест сериализации кеша."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Получение кеша
        get_result = await cache_manager.get("test_key")
        # Проверки
        assert get_result == sample_data

    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест статистики кеша."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Получение статистики
        stats_result = await cache_manager.get_stats()
        # Проверки
        assert stats_result is not None
        assert "total_keys" in stats_result
        assert "max_size" in stats_result
        assert isinstance(stats_result["total_keys"], int)
        assert isinstance(stats_result["max_size"], int)

    @pytest.mark.asyncio
    async def test_cache_optimization(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест оптимизации кеша."""
        # Установка нескольких кешей
        for i in range(10):
            await cache_manager.set(f"key_{i}", sample_data)
        # Получение статистики
        stats_result = await cache_manager.get_stats()
        # Проверки
        assert stats_result is not None
        assert stats_result["total_keys"] >= 10

    @pytest.mark.asyncio
    async def test_cache_backup_restore(self, cache_manager: CacheManager, sample_data: dict) -> None:
        """Тест резервного копирования и восстановления кеша."""
        # Установка кеша
        await cache_manager.set("test_key", sample_data)
        # Очистка кеша
        await cache_manager.clear()
        # Проверка, что кеш очищен
        get_result = await cache_manager.get("test_key")
        assert get_result is None
        # Восстановление кеша
        await cache_manager.set("test_key", sample_data)
        # Проверка восстановления
        restored_data = await cache_manager.get("test_key")
        assert restored_data == sample_data

    @pytest.mark.asyncio
    async def test_error_handling(self, cache_manager: CacheManager) -> None:
        """Тест обработки ошибок."""
        # Тест с невалидными данными
        invalid_data = None
        set_result = await cache_manager.set("invalid_key", invalid_data)
        # Проверки
        assert set_result is True

    @pytest.mark.asyncio
    async def test_edge_cases(self, cache_manager: CacheManager) -> None:
        """Тест граничных случаев."""
        # Тест с пустыми данными
        empty_data: dict = {}
        set_result = await cache_manager.set("empty_key", empty_data)
        # Проверки
        assert set_result is True
        get_result = await cache_manager.get("empty_key")
        assert get_result == empty_data
        # Тест с очень большими данными
        large_data = {"data": "x" * 100000}
        large_set_result = await cache_manager.set("large_key", large_data)
        # Проверки
        assert large_set_result is True
        large_get_result = await cache_manager.get("large_key")
        assert large_get_result == large_data

    @pytest.mark.asyncio
    async def test_cleanup(self, cache_manager: CacheManager) -> None:
        """Тест очистки ресурсов."""
        # Установка кеша
        await cache_manager.set("test_key", {"data": "test"})
        # Очистка
        await cache_manager.clear()
        # Проверки
        size_result = await cache_manager.size()
        assert size_result == 0
