"""
Cache management module for shared components.
"""

import asyncio
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set
from datetime import datetime, timedelta
import weakref
import threading
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Статистика кэша."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Коэффициент попаданий в кэш."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@dataclass
class CacheEntry:
    """Запись в кэше."""
    value: Any
    created_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Проверка истечения срока жизни."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Обновление времени последнего доступа."""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheProtocol(ABC):
    """Протокол для кэша."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения по ключу."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения по ключу."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Удаление значения по ключу."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Очистка всего кэша."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Получение размера кэша."""
        pass

class MemoryCache(CacheProtocol):
    """In-memory реализация кэша с поддержкой TTL и LRU."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self._storage: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
        
        # Фоновый процесс будет запущен явно через start_background_task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_background_task(self) -> None:
        """Запуск фонового процесса очистки."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_background_task(self) -> None:
        """Остановка фонового процесса очистки."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            finally:
                self._cleanup_task = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения по ключу."""
        async with self._lock:
            entry = self._storage.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            
            if entry.is_expired():
                del self._storage[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения по ключу."""
        try:
            async with self._lock:
                # Проверка лимита размера и выселение старых записей
                if len(self._storage) >= self._max_size and key not in self._storage:
                    await self._evict_lru()
                
                ttl_to_use = ttl if ttl is not None else self._default_ttl
                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    ttl=ttl_to_use
                )
                
                self._storage[key] = entry
                self._stats.sets += 1
                return True
        except Exception as e:
            logger.error(f"Ошибка при установке значения в кэш: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление значения по ключу."""
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                self._stats.deletes += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        async with self._lock:
            entry = self._storage.get(key)
            if entry is None:
                return False
            
            if entry.is_expired():
                del self._storage[key]
                self._stats.evictions += 1
                return False
            
            return True

    async def clear(self) -> None:
        """Очистка всего кэша."""
        async with self._lock:
            self._storage.clear()
            self._stats = CacheStats()

    async def size(self) -> int:
        """Получение размера кэша."""
        async with self._lock:
            return len(self._storage)
    
    async def get_stats(self) -> CacheStats:
        """Получение статистики кэша."""
        return self._stats
    
    async def _evict_lru(self) -> None:
        """Выселение наименее используемых записей."""
        if not self._storage:
            return
        
        # Сортируем по времени последнего доступа
        sorted_items = sorted(
            self._storage.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Удаляем 20% наименее используемых записей
        evict_count = max(1, len(sorted_items) // 5)
        for i in range(evict_count):
            key = sorted_items[i][0]
            del self._storage[key]
            self._stats.evictions += 1
    
    async def _cleanup_loop(self) -> None:
        """Фоновый процесс очистки истёкших записей."""
        while True:
            try:
                await asyncio.sleep(60)  # Проверка каждую минуту
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в процессе очистки кэша: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Очистка истёкших записей."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._storage.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._storage[key]
                self._stats.evictions += 1

class PersistentCache(MemoryCache):
    """Персистентный кэш с сохранением на диск."""
    
    def __init__(self, cache_file: str, max_size: int = 1000, default_ttl: Optional[int] = None):
        super().__init__(max_size, default_ttl)
        self._cache_file = cache_file
        asyncio.create_task(self._load_from_disk())
    
    async def _load_from_disk(self) -> None:
        """Загрузка кэша с диска."""
        try:
            import os
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self._storage = data.get('storage', {})
                    self._stats = data.get('stats', CacheStats())
                    logger.info(f"Кэш загружен с диска: {len(self._storage)} записей")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша с диска: {e}")
    
    async def _save_to_disk(self) -> None:
        """Сохранение кэша на диск."""
        try:
            data = {
                'storage': self._storage,
                'stats': self._stats,
                'saved_at': time.time()
            }
            with open(self._cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша на диск: {e}")
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения с сохранением на диск."""
        result = await super().set(key, value, ttl)
        if result:
            await self._save_to_disk()
        return result
    
    async def delete(self, key: str) -> bool:
        """Удаление значения с сохранением на диск."""
        result = await super().delete(key)
        if result:
            await self._save_to_disk()
        return result

class CacheManager:
    """Менеджер кэшей с поддержкой нескольких уровней."""
    
    def __init__(self) -> None:
        self._caches: Dict[str, CacheProtocol] = {}
        self._default_cache: Optional[CacheProtocol] = None
    
    def register_cache(self, name: str, cache: CacheProtocol, is_default: bool = False) -> None:
        """Регистрация кэша."""
        self._caches[name] = cache
        if is_default or self._default_cache is None:
            self._default_cache = cache
    
    def get_cache(self, name: Optional[str] = None) -> Optional[CacheProtocol]:
        """Получение кэша по имени."""
        if name is None:
            return self._default_cache
        return self._caches.get(name)
    
    async def get(self, key: str, cache_name: Optional[str] = None) -> Optional[Any]:
        """Получение значения из кэша."""
        cache = self.get_cache(cache_name)
        if cache is None:
            return None
        return await cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_name: Optional[str] = None) -> bool:
        """Установка значения в кэш."""
        cache = self.get_cache(cache_name)
        if cache is None:
            return False
        return await cache.set(key, value, ttl)

# Глобальный менеджер кэшей
cache_manager = CacheManager()

# Создание кэша по умолчанию
default_cache = MemoryCache(max_size=10000, default_ttl=3600)
cache_manager.register_cache("default", default_cache, is_default=True)

def get_cache_manager() -> CacheManager:
    """Получение глобального менеджера кэшей."""
    return cache_manager

# Декоратор для кэширования результатов функций
def cached(ttl: Optional[int] = None, cache_name: Optional[str] = None):
    """Декоратор для кэширования результатов функций."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Создание ключа кэша
            cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Попытка получить из кэша
            cached_result = await cache_manager.get(cache_key, cache_name)
            if cached_result is not None:
                return cached_result
            
            # Выполнение функции и кэширование результата
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, cache_name)
            return result
        
        return wrapper
    return decorator
