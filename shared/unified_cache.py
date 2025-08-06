"""
Унифицированная система кэширования для ATB.
"""

import asyncio
import hashlib
import json
import pickle
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
from shared.numpy_utils import np
import time

from loguru import logger


class CacheEntry:
    """Запись в кэше."""

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        created_at: Optional[datetime] = None,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or datetime.now()
        self.access_count = access_count
        self.last_accessed = last_accessed or datetime.now()
        self.metadata = metadata or {}

    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)

    def access(self) -> None:
        """Отметить доступ к записи."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def get_age(self) -> float:
        """Получить возраст записи в секундах."""
        return (datetime.now() - self.created_at).total_seconds()

    def get_size(self) -> int:
        """Получить размер записи в байтах."""
        try:
            if isinstance(self.value, pd.DataFrame):
                # Проверяем наличие метода memory_usage
                if hasattr(self.value, 'memory_usage'):
                    return int(self.value.memory_usage(deep=True).sum())
                else:
                    # Альтернативный способ оценки размера DataFrame
                    return int(len(pickle.dumps(self.value)))
            elif isinstance(self.value, np.ndarray):
                return int(self.value.nbytes)
            else:
                return int(len(pickle.dumps(self.value)))
        except Exception:
            return 1024  # Примерный размер по умолчанию


class CachePolicy:
    """Политика кэширования."""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = "lru"  # lru, lfu, fifo, random

    def should_evict(self, cache: Dict[str, CacheEntry]) -> bool:
        """Проверить, нужно ли удалить записи."""
        if len(cache) > self.max_size:
            return True
        total_memory = sum(entry.get_size() for entry in cache.values())
        if total_memory > self.max_memory_bytes:
            return True
        return False

    def select_eviction_candidates(
        self, cache: Dict[str, CacheEntry], count: int = 1
    ) -> List[str]:
        """Выбрать кандидатов для удаления."""
        if self.eviction_policy == "lru":
            # Least Recently Used
            sorted_entries = sorted(cache.items(), key=lambda x: x[1].last_accessed)
        elif self.eviction_policy == "lfu":
            # Least Frequently Used
            sorted_entries = sorted(cache.items(), key=lambda x: x[1].access_count)
        elif self.eviction_policy == "fifo":
            # First In First Out
            sorted_entries = sorted(cache.items(), key=lambda x: x[1].created_at)
        else:  # random
            import random

            keys = list(cache.keys())
            return random.sample(keys, min(count, len(keys)))
        return [entry[0] for entry in sorted_entries[:count]]


class UnifiedCache:
    """Унифицированная система кэширования."""

    def __init__(
        self,
        cache_dir: str = "cache",
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[int] = 3600,
        enable_persistence: bool = True,
        compression: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.policy = CachePolicy(max_size, max_memory_mb)
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.compression = compression
        # In-memory кэш
        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        # Статистика
        self.stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0,
        }
        # Загрузка сохраненного кэша
        if self.enable_persistence:
            self._load_persistent_cache()
        # Запуск фоновых задач
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Запуск фоновых задач."""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._persistence_task = asyncio.create_task(self._periodic_persistence())

    async def _periodic_cleanup(self) -> None:
        """Периодическая очистка кэша с адаптивными интервалами."""
        base_interval = 300  # 5 минут базовый интервал
        current_interval = base_interval
        consecutive_errors = 0
        
        while True:
            try:
                await asyncio.sleep(current_interval)
                
                # Адаптивная логика: увеличиваем интервал при низкой нагрузке
                cache_size = len(self._cache)
                if cache_size < 100:
                    current_interval = min(base_interval * 2, 600)  # До 10 минут
                elif cache_size > 500:
                    current_interval = max(base_interval // 2, 60)  # Минимум 1 минута
                else:
                    current_interval = base_interval
                
                expired_count = await self._cleanup_expired()
                logger.debug(f"Cleaned {expired_count} expired cache entries")
                
                # Сбрасываем счетчик ошибок при успешном выполнении
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                # Экспоненциальная задержка при ошибках
                error_delay = min(60 * (2 ** consecutive_errors), 1800)  # Максимум 30 минут
                logger.error(f"Error in periodic cleanup (attempt {consecutive_errors}): {e}")
                await asyncio.sleep(error_delay)

    async def _periodic_persistence(self) -> None:
        """Периодическое сохранение кэша с умным мониторингом."""
        base_interval = 600  # 10 минут базовый интервал
        current_interval = base_interval
        consecutive_errors = 0
        last_save_time = 0
        
        while True:
            try:
                await asyncio.sleep(current_interval)
                
                # Проверяем, есть ли изменения для сохранения
                current_time = time.time()
                cache_size = len(self._cache)
                
                # Адаптивные интервалы на основе активности
                if cache_size == 0:
                    current_interval = base_interval * 3  # Увеличиваем при пустом кэше
                elif cache_size > 1000:
                    current_interval = base_interval // 2  # Чаще сохраняем при большом кэше
                else:
                    current_interval = base_interval
                
                # Сохраняем только если прошло достаточно времени с последнего сохранения
                if current_time - last_save_time >= 300:  # Минимум 5 минут между сохранениями
                    await self._save_persistent_cache()
                    last_save_time = int(current_time)
                    logger.debug(f"Persisted cache with {cache_size} entries")
                
                # Сбрасываем счетчик ошибок при успешном выполнении
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                # Экспоненциальная задержка при ошибках
                error_delay = min(120 * (2 ** consecutive_errors), 3600)  # Максимум 1 час
                logger.error(f"Error in periodic persistence (attempt {consecutive_errors}): {e}")
                await asyncio.sleep(error_delay)

    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Генерация ключа кэша."""
        key_data: Dict[str, Any] = {"args": args, "kwargs": sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение из кэша."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    self.stats["misses"] += 1
                    return default
                entry.access()
                self.stats["hits"] += 1
                # Перемещаем в конец (LRU)
                if hasattr(self._cache, 'move_to_end'):
                    self._cache.move_to_end(key)
                return entry.value
            else:
                self.stats["misses"] += 1
                return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Установить значение в кэш."""
        with self._lock:
            # Проверяем, нужно ли удалить записи
            if self.policy.should_evict(self._cache):
                self._evict_entries()
            # Создаем новую запись
            entry = CacheEntry(
                key=key, value=value, ttl=ttl or self.default_ttl, metadata=metadata
            )
            # Добавляем в кэш
            self._cache[key] = entry
            # Обновляем статистику
            self.stats["size"] = len(self._cache)
            self.stats["memory_usage"] = sum(
                entry.get_size() for entry in self._cache.values()
            )

    def _evict_entries(self, count: int = 1) -> None:
        """Удалить записи из кэша."""
        candidates = self.policy.select_eviction_candidates(self._cache, count)
        for key in candidates:
            if key in self._cache:
                del self._cache[key]
                self.stats["evictions"] += 1
        # Обновляем статистику
        self.stats["size"] = len(self._cache)
        self.stats["memory_usage"] = sum(
            entry.get_size() for entry in self._cache.values()
        )

    async def _cleanup_expired(self) -> int:
        """Очистка истекших записей."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def exists(self, key: str) -> bool:
        """Проверить существование ключа."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    return False
                return True
            return False

    def delete(self, key: str) -> bool:
        """Удалить запись из кэша."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Очистить весь кэш."""
        with self._lock:
            self._cache.clear()
            self.stats["size"] = 0
            self.stats["memory_usage"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        with self._lock:
            hit_rate = (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0
            )
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "entries": len(self._cache),
                "avg_entry_size": (
                    self.stats["memory_usage"] / len(self._cache)
                    if len(self._cache) > 0
                    else 0
                ),
            }

    def _get_cache_file_path(self) -> Path:
        """Получить путь к файлу кэша."""
        return self.cache_dir / "unified_cache.pkl"

    async def _save_persistent_cache(self) -> None:
        """Сохранить кэш на диск."""
        if not self.enable_persistence:
            return
        try:
            cache_data: Dict[str, Any] = {
                "entries": {},
                "stats": self.stats,
                "timestamp": datetime.now().isoformat(),
            }
            # Сохраняем только неистекшие записи
            with self._lock:
                for key, entry in self._cache.items():
                    if not entry.is_expired():
                        cache_data["entries"][key] = {
                            "value": entry.value,
                            "ttl": entry.ttl,
                            "created_at": entry.created_at.isoformat(),
                            "access_count": entry.access_count,
                            "last_accessed": entry.last_accessed.isoformat(),
                            "metadata": entry.metadata,
                        }
            cache_file = self._get_cache_file_path()
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            logger.debug(
                f"Saved persistent cache: {len(cache_data['entries'])} entries"
            )
        except Exception as e:
            logger.error(f"Error saving persistent cache: {e}")

    def _load_persistent_cache(self) -> None:
        """Загрузить кэш с диска."""
        if not self.enable_persistence:
            return
        try:
            cache_file = self._get_cache_file_path()
            if not cache_file.exists():
                return
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            # Восстанавливаем записи
            for key, entry_data in cache_data.get("entries", {}).items():
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    ttl=entry_data["ttl"],
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    access_count=entry_data["access_count"],
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    metadata=entry_data["metadata"],
                )
                # Проверяем, не истекла ли запись
                if not entry.is_expired():
                    self._cache[key] = entry
            # Восстанавливаем статистику
            self.stats.update(cache_data.get("stats", {}))
            logger.info(f"Loaded persistent cache: {len(self._cache)} entries")
        except Exception as e:
            logger.error(f"Error loading persistent cache: {e}")


# Глобальный экземпляр кэша
_cache_instance: Optional[UnifiedCache] = None
_cache_lock = threading.Lock()


def get_cache_manager() -> UnifiedCache:
    """Получить глобальный менеджер кэша."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = UnifiedCache()
    return _cache_instance


def cache_result(ttl: Optional[int] = None, key_prefix: str = "") -> Callable[[Callable], Callable]:
    """Декоратор для кэширования результатов функций."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache_manager()
            # Генерируем ключ
            func_key = f"{key_prefix}:{func.__name__}"
            cache_key = cache._generate_key(func_key, *args, **kwargs)
            # Пытаемся получить из кэша
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            # Выполняем функцию
            result = await func(*args, **kwargs)
            # Сохраняем в кэш
            cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator


def cache_sync_result(ttl: Optional[int] = None, key_prefix: str = "") -> Callable[[Callable], Callable]:
    """Декоратор для кэширования результатов синхронных функций."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache_manager()
            # Генерируем ключ
            func_key = f"{key_prefix}:{func.__name__}"
            cache_key = cache._generate_key(func_key, *args, **kwargs)
            # Пытаемся получить из кэша
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            # Выполняем функцию
            result = func(*args, **kwargs)
            # Сохраняем в кэш
            cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator
