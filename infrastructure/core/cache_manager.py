# -*- coding: utf-8 -*-
"""Менеджер кэша для infrastructure слоя."""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from shared.logging import LoggerMixin


class CacheManager(LoggerMixin):
    """Менеджер кэширования данных."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        super().__init__()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.expiration_times: Dict[str, datetime] = {}

    async def get(self, key: str, default: Any = None) -> Any:
        """Получение значения из кэша."""
        if key not in self.cache:
            return default
        # Проверка срока действия
        if self._is_expired(key):
            await self.delete(key)
            return default
        # Обновление времени доступа
        self.access_times[key] = datetime.now()
        self.log_debug(f"Cache hit for key: {key}")
        return self.cache[key]["value"]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
        try:
            # Проверка размера кэша
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()
            # Сохранение значения
            self.cache[key] = {"value": value, "created_at": datetime.now()}
            # Установка времени истечения
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            self.expiration_times[key] = datetime.now() + timedelta(seconds=ttl_seconds)
            self.access_times[key] = datetime.now()
            self.log_debug(f"Cache set for key: {key}, TTL: {ttl_seconds}s")
            return True
        except Exception as e:
            self.log_error(f"Failed to set cache for key {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        try:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.expiration_times:
                del self.expiration_times[key]
            self.log_debug(f"Cache deleted for key: {key}")
            return True
        except Exception as e:
            self.log_error(f"Failed to delete cache for key {key}: {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа в кэше."""
        if key not in self.cache:
            return False
        if self._is_expired(key):
            await self.delete(key)
            return False
        return True

    async def expire(self, key: str, ttl: int) -> bool:
        """Установка времени истечения для ключа."""
        if key not in self.cache:
            return False
        self.expiration_times[key] = datetime.now() + timedelta(seconds=ttl)
        return True

    async def ttl(self, key: str) -> int:
        """Получение оставшегося времени жизни ключа."""
        if key not in self.cache or self._is_expired(key):
            return -1
        remaining = self.expiration_times[key] - datetime.now()
        return max(0, int(remaining.total_seconds()))

    async def clear(self) -> bool:
        """Очистка всего кэша."""
        try:
            self.cache.clear()
            self.access_times.clear()
            self.expiration_times.clear()
            self.log_info("Cache cleared")
            return True
        except Exception as e:
            self.log_error(f"Failed to clear cache: {str(e)}")
            return False

    async def size(self) -> int:
        """Получение размера кэша."""
        return len(self.cache)

    async def keys(self, pattern: str = "*") -> list:
        """Получение списка ключей."""
        # Простая реализация паттерна
        if pattern == "*":
            return list(self.cache.keys())
        # Базовая поддержка wildcard
        keys = []
        for key in self.cache.keys():
            if self._match_pattern(key, pattern):
                keys.append(key)
        return keys

    async def get_many(self, keys: list) -> Dict[str, Any]:
        """Получение множества значений."""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Установка множества значений."""
        try:
            for key, value in data.items():
                await self.set(key, value, ttl)
            return True
        except Exception as e:
            self.log_error(f"Failed to set many cache entries: {str(e)}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Инкремент числового значения."""
        current_value = await self.get(key, 0)
        if isinstance(current_value, (int, float)):
            new_value = current_value + amount
            await self.set(key, new_value)
            return int(new_value)
        return None

    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Декремент числового значения."""
        return await self.increment(key, -amount)

    async def get_or_set(
        self, key: str, default_value: Any, ttl: Optional[int] = None
    ) -> Any:
        """Получение значения или установка по умолчанию."""
        value = await self.get(key)
        if value is None:
            await self.set(key, default_value, ttl)
            return default_value
        return value

    async def set_if_not_exists(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Установка значения только если ключ не существует."""
        if await self.exists(key):
            return False
        return await self.set(key, value, ttl)

    def _is_expired(self, key: str) -> bool:
        """Проверка истечения срока действия ключа."""
        if key not in self.expiration_times:
            return False
        return datetime.now() > self.expiration_times[key]

    async def _evict_oldest(self) -> None:
        """Удаление самого старого элемента (LRU)."""
        if not self.access_times:
            return
        # Находим самый старый элемент
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self.delete(oldest_key)

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Простое сопоставление с паттерном."""
        if pattern == "*":
            return True
        # Базовая поддержка wildcard
        if "*" in pattern:
            parts = pattern.split("*")
            if len(parts) == 2:
                return key.startswith(parts[0]) and key.endswith(parts[1])
        return key == pattern

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        total_keys = len(self.cache)
        expired_keys = sum(1 for key in self.cache if self._is_expired(key))
        return {
            "total_keys": total_keys,
            "expired_keys": expired_keys,
            "active_keys": total_keys - expired_keys,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }

    async def cleanup_expired(self) -> Optional[int]:
        """Очистка истекших ключей."""
        expired_keys = []
        for key in self.cache.keys():
            if self._is_expired(key):
                expired_keys.append(key)
        for key in expired_keys:
            await self.delete(key)
        self.log_info(f"Cleaned up {len(expired_keys)} expired keys")
        return int(len(expired_keys)) if expired_keys else None
