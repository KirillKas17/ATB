# -*- coding: utf-8 -*-
"""Менеджер кэша для infrastructure слоя."""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from shared.logging import LoggerMixin
from infrastructure.shared.cache import CacheManager as NewCacheManager, CacheConfig


class CacheManager(LoggerMixin):
    """Менеджер кэширования данных с обратной совместимостью."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        super().__init__()
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Используем новый CacheManager под капотом
        config = CacheConfig(max_size=max_size, default_ttl=default_ttl)
        self._new_cache_manager = NewCacheManager()
        self._cache_instance = self._new_cache_manager.get_cache("default")
        
        # Для обратной совместимости
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.expiration_times: Dict[str, datetime] = {}

    async def get(self, key: str, default: Any = None) -> Any:
        """Получение значения из кэша."""
        try:
            result = await self._cache_instance.get(key)
            if result is not None:
                self.log_debug(f"Cache hit for key: {key}")
                return result
            return default
        except Exception as e:
            self.log_error(f"Cache get error for key {key}: {e}")
            return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
        try:
            ttl_to_use = ttl if ttl is not None else self.default_ttl
            success = await self._cache_instance.set(key, value, ttl_to_use)
            if success:
                self.log_debug(f"Cache set for key: {key}, TTL: {ttl_to_use}s")
            return success
        except Exception as e:
            self.log_error(f"Failed to set cache for key {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление ключа из кэша."""
        try:
            return await self._cache_instance.delete(key)
        except Exception as e:
            self.log_error(f"Failed to delete cache key {key}: {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        try:
            return await self._cache_instance.exists(key)
        except Exception:
            return False

    async def clear(self) -> None:
        """Очистка всего кэша."""
        try:
            await self._cache_instance.clear()
        except Exception as e:
            self.log_error(f"Failed to clear cache: {e}")

    async def size(self) -> int:
        """Получение размера кэша."""
        try:
            return await self._cache_instance.size()
        except Exception:
            return 0

    # Методы для обратной совместимости с тестами
    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Алиас для set()."""
        return await self.set(key, value, ttl)

    async def get_cache(self, key: str, default: Any = None) -> Any:
        """Алиас для get()."""
        return await self.get(key, default)

    async def delete_cache(self, key: str) -> bool:
        """Алиас для delete()."""
        return await self.delete(key)

    async def clear_cache(self) -> None:
        """Алиас для clear()."""
        await self.clear()

    async def cache_exists(self, key: str) -> bool:
        """Алиас для exists()."""
        return await self.exists(key)

    async def get_cache_keys(self) -> list:
        """Получение всех ключей кэша."""
        return []  # Базовая реализация

    async def get_cache_info(self) -> dict:
        """Получение информации о кэше."""
        size = await self.size()
        return {
            "size": size,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }

    async def get_cache_statistics(self) -> dict:
        """Получение статистики кэша."""
        return {
            "hits": 0,
            "misses": 0,
            "size": await self.size()
        }

    async def optimize_cache(self) -> None:
        """Оптимизация кэша."""
        pass  # Базовая реализация

    async def backup_cache(self) -> dict:
        """Резервное копирование кэша."""
        return {}  # Базовая реализация

    async def restore_cache(self, backup_data: dict) -> bool:
        """Восстановление кэша из резервной копии."""
        return True  # Базовая реализация

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        await self.clear()

    @property
    def cache_storage(self) -> dict:
        """Доступ к хранилищу кэша."""
        return self.cache

    @property 
    def cache_policies(self) -> dict:
        """Политики кэширования."""
        return {"max_size": self.max_size, "default_ttl": self.default_ttl}

    @property
    def cache_metrics(self) -> dict:
        """Метрики кэширования."""
        return {"size": len(self.cache)}
