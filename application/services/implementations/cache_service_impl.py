"""
Промышленная реализация CacheService.
"""

import asyncio
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from application.protocols.service_protocols import CacheService
from application.services.base_service import BaseApplicationService


class CacheServiceImpl(BaseApplicationService, CacheService):
    """Промышленная реализация сервиса кэширования."""

    def __init__(self, *args, **kwargs) -> Any:
        super().__init__("CacheService", config)
        # Кэш хранилище
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Конфигурация
        self.default_ttl = self.config.get("default_ttl", 300)  # 5 минут
        self.max_size = self.config.get("max_size", 10000)
        self.cleanup_interval = self.config.get("cleanup_interval", 60)  # секунды
        self.eviction_policy = self.config.get(
            "eviction_policy", "LRU"
        )  # LRU, LFU, FIFO
        # Статистика
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "total_size_bytes": 0,
        }
        # Фоновые задачи
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Инициализация сервиса."""
        # Запускаем фоновые задачи
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._stats_task = asyncio.create_task(self._stats_loop())
        self.logger.info("CacheService initialized")

    async def validate_config(self) -> bool:
        """Валидация конфигурации."""
        required_configs = [
            "default_ttl",
            "max_size",
            "cleanup_interval",
            "eviction_policy",
        ]
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        return await self._execute_with_metrics("get", self._get_impl, key)

    async def _get_impl(self, key: str) -> Optional[Any]:
        """Реализация получения значения из кэша."""
        try:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            entry = self._cache[key]
            # Проверяем истечение срока действия
            if self._is_expired(entry):
                await self.delete(key)
                self._stats["misses"] += 1
                return None
            # Обновляем время последнего доступа
            entry["last_accessed"] = datetime.now()
            entry["access_count"] += 1
            # Десериализуем значение
            value = self._deserialize(entry["value"], entry["serialization_type"])
            self._stats["hits"] += 1
            return value
        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            self._stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
        return await self._execute_with_metrics("set", self._set_impl, key, value, ttl)

    async def _set_impl(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Реализация установки значения в кэш."""
        try:
            # Проверяем размер кэша
            if len(self._cache) >= self.max_size:
                await self._evict_entries()
            # Сериализуем значение
            serialized_value, serialization_type = self._serialize(value)
            # Создаем запись кэша
            entry = {
                "value": serialized_value,
                "serialization_type": serialization_type,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 0,
                "ttl": ttl or self.default_ttl,
                "size_bytes": (
                    len(serialized_value)
                    if isinstance(serialized_value, bytes)
                    else len(str(serialized_value))
                ),
            }
            # Обновляем статистику размера
            if key in self._cache:
                old_size = self._cache[key]["size_bytes"]
                self._stats["total_size_bytes"] -= old_size
            self._cache[key] = entry
            self._stats["total_size_bytes"] += entry["size_bytes"]
            self._stats["sets"] += 1
            return True
        except Exception as e:
            self.logger.error(f"Error setting cache value: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        return await self._execute_with_metrics("delete", self._delete_impl, key)

    async def _delete_impl(self, key: str) -> bool:
        """Реализация удаления значения из кэша."""
        try:
            if key in self._cache:
                # Обновляем статистику размера
                old_size = self._cache[key]["size_bytes"]
                self._stats["total_size_bytes"] -= old_size
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {e}")
            return False

    async def clear(self, pattern: str = "*") -> bool:
        """Очистка кэша по паттерну."""
        return await self._execute_with_metrics("clear", self._clear_impl, pattern)

    async def _clear_impl(self, pattern: str = "*") -> bool:
        """Реализация очистки кэша по паттерну."""
        try:
            if pattern == "*":
                # Очищаем весь кэш
                self._cache.clear()
                self._stats["total_size_bytes"] = 0
                self._stats["deletes"] += len(self._cache)
            else:
                # Очищаем по паттерну
                keys_to_delete = []
                for key in self._cache.keys():
                    if self._matches_pattern(key, pattern):
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    await self.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        return await self._execute_with_metrics("get_stats", self._get_stats_impl)

    async def _get_stats_impl(self) -> Dict[str, Any]:
        """Реализация получения статистики кэша."""
        try:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            )
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "miss_rate": 1.0 - hit_rate,
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "evictions": self._stats["evictions"],
                "total_entries": len(self._cache),
                "total_size_bytes": self._stats["total_size_bytes"],
                "max_size": self.max_size,
                "eviction_policy": self.eviction_policy,
                "default_ttl": self.default_ttl,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}  

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        return await self._execute_with_metrics("exists", self._exists_impl, key)

    async def _exists_impl(self, key: str) -> bool:
        """Реализация проверки существования ключа."""
        try:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            # Проверяем истечение срока действия
            if self._is_expired(entry):
                await self.delete(key)
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking key existence: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Установка времени жизни ключа."""
        return await self._execute_with_metrics("expire", self._expire_impl, key, ttl)

    async def _expire_impl(self, key: str, ttl: int) -> bool:
        """Реализация установки времени жизни ключа."""
        try:
            if key in self._cache:
                self._cache[key]["ttl"] = ttl
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting key expiration: {e}")
            return False

    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Получение нескольких значений из кэша."""
        return await self._execute_with_metrics("get_multi", self._get_multi_impl, keys)

    async def _get_multi_impl(self, keys: List[str]) -> Dict[str, Any]:
        """Реализация получения нескольких значений из кэша."""
        try:
            result = {}
            for key in keys:
                value = await self.get(key)
                if value is not None:
                    result[key] = value
            return result
        except Exception as e:
            self.logger.error(f"Error getting multiple values from cache: {e}")
            return {}  

    async def set_multi(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Установка нескольких значений в кэш."""
        return await self._execute_with_metrics(
            "set_multi", self._set_multi_impl, data, ttl
        )

    async def _set_multi_impl(
        self, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Реализация установки нескольких значений в кэш."""
        try:
            success_count = 0
            for key, value in data.items():
                success = await self.set(key, value, ttl)
                if success:
                    success_count += 1
            return success_count == len(data)
        except Exception as e:
            self.logger.error(f"Error setting multiple values in cache: {e}")
            return False

    async def delete_multi(self, keys: List[str]) -> bool:
        """Удаление нескольких значений из кэша."""
        return await self._execute_with_metrics(
            "delete_multi", self._delete_multi_impl, keys
        )

    async def _delete_multi_impl(self, keys: List[str]) -> bool:
        """Реализация удаления нескольких значений из кэша."""
        try:
            success_count = 0
            for key in keys:
                success = await self.delete(key)
                if success:
                    success_count += 1
            return success_count == len(keys)
        except Exception as e:
            self.logger.error(f"Error deleting multiple values from cache: {e}")
            return False

    def generate_key(self, *args, **kwargs) -> str:
        """Генерация ключа кэша."""
        try:
            # Создаем строку из аргументов
            key_parts = []
            for arg in args:
                key_parts.append(str(arg))
            for key, value in sorted(kwargs.items()):
                key_parts.append(f"{key}:{value}")
            key_string = "|".join(key_parts)
            # Создаем хеш ключа
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"cache:{key_hash}"
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return f"cache:error:{datetime.now().timestamp()}"

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Проверка истечения срока действия записи."""
        try:
            if entry["ttl"] is None:
                return False
            expiration_time = entry["created_at"] + timedelta(seconds=entry["ttl"])
            return datetime.now() > expiration_time
        except Exception as e:
            self.logger.error(f"Error checking expiration: {e}")
            return True

    def _serialize(self, value: Any) -> tuple[Any, str]:
        """Сериализация значения."""
        try:
            # Пробуем JSON для простых типов
            try:
                json_value = json.dumps(value, default=str)
                return json_value, "json"
            except (TypeError, ValueError):
                pass
            # Используем pickle для сложных объектов
            try:
                pickle_value = pickle.dumps(value)
                return pickle_value, "pickle"
            except (TypeError, ValueError):
                pass
            # Fallback к строке
            return str(value), "string"
        except Exception as e:
            self.logger.error(f"Error serializing value: {e}")
            return str(value), "string"

    def _deserialize(self, value: Any, serialization_type: str) -> Any:
        """Десериализация значения."""
        try:
            if serialization_type == "json":
                return json.loads(value)
            elif serialization_type == "pickle":
                return pickle.loads(value)
            elif serialization_type == "string":
                return value
            else:
                return value
        except Exception as e:
            self.logger.error(f"Error deserializing value: {e}")
            return value

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Проверка соответствия ключа паттерну."""
        try:
            if pattern == "*":
                return True
            # Простая проверка по подстроке
            return pattern in key
        except Exception as e:
            self.logger.error(f"Error matching pattern: {e}")
            return False

    async def _evict_entries(self) -> None:
        """Вытеснение записей из кэша."""
        try:
            if len(self._cache) < self.max_size:
                return
            # Удаляем истекшие записи
            expired_keys = []
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            for key in expired_keys:
                await self.delete(key)
            # Если все еще превышаем лимит, применяем политику вытеснения
            if len(self._cache) >= self.max_size:
                keys_to_evict = self._select_keys_for_eviction()
                for key in keys_to_evict:
                    await self.delete(key)
                    self._stats["evictions"] += 1
        except Exception as e:
            self.logger.error(f"Error evicting entries: {e}")

    def _select_keys_for_eviction(self) -> List[str]:
        """Выбор ключей для вытеснения."""
        try:
            if self.eviction_policy == "LRU":
                # Least Recently Used
                sorted_entries = sorted(
                    self._cache.items(), key=lambda x: x[1]["last_accessed"]
                )
            elif self.eviction_policy == "LFU":
                # Least Frequently Used
                sorted_entries = sorted(
                    self._cache.items(), key=lambda x: x[1]["access_count"]
                )
            elif self.eviction_policy == "FIFO":
                # First In First Out
                sorted_entries = sorted(
                    self._cache.items(), key=lambda x: x[1]["created_at"]
                )
            else:
                # По умолчанию LRU
                sorted_entries = sorted(
                    self._cache.items(), key=lambda x: x[1]["last_accessed"]
                )
            # Выбираем 10% самых старых/менее используемых записей
            evict_count = max(1, len(sorted_entries) // 10)
            return [key for key, _ in sorted_entries[:evict_count]]
        except Exception as e:
            self.logger.error(f"Error selecting keys for eviction: {e}")
            return []  

    async def _cleanup_loop(self) -> None:
        """Цикл очистки кэша."""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                # Удаляем истекшие записи
                expired_keys = []
                for key, entry in self._cache.items():
                    if self._is_expired(entry):
                        expired_keys.append(key)
                for key in expired_keys:
                    await self.delete(key)
                # Проверяем размер кэша
                if len(self._cache) > self.max_size:
                    await self._evict_entries()
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _stats_loop(self) -> None:
        """Цикл сбора статистики."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Каждую минуту
                # Логируем статистику
                stats = await self.get_stats()
                if stats:
                    self.logger.debug(f"Cache stats: {stats}")
            except Exception as e:
                self.logger.error(f"Error in stats loop: {e}")

    async def stop(self) -> None:
        """Остановка сервиса."""
        await super().stop()
        # Отменяем фоновые задачи
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._stats_task:
            self._stats_task.cancel()
        # Очищаем кэш
        self._cache.clear()
        self._stats["total_size_bytes"] = 0

    # Реализация абстрактных методов из BaseService
    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных для кэш-операций."""
        if isinstance(data, dict):
            if "operation" in data:
                return data["operation"] in ["get", "set", "delete", "clear", "exists"]
            return True
        return False

    def process(self, data: Any) -> Any:
        """Обработка кэш-данных."""
        if not self.validate_input(data):
            return {"error": "Invalid input data"}
        
        try:
            if isinstance(data, dict) and "operation" in data:
                operation = data["operation"]
                return {"status": f"{operation}_processing", "data": data}
            
            return {"status": "processed", "data": data}
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
