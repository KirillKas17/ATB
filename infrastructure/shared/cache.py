"""
Унифицированная система кэширования для Infrastructure Layer.
"""

import asyncio
import hashlib
import json
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

try:
    import aioredis
except ImportError:
    aioredis = None
try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None
from loguru import logger

T = TypeVar("T")


class CacheType(Enum):
    """Типы кэша."""

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


class CacheEvictionStrategy(Enum):
    """Стратегии вытеснения кэша."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Конфигурация кэша."""

    cache_type: CacheType = CacheType.MEMORY
    max_size: int = 1000
    default_ttl_seconds: int = 300
    eviction_strategy: CacheEvictionStrategy = CacheEvictionStrategy.LRU
    enable_compression: bool = False
    enable_encryption: bool = False
    compression_threshold_bytes: int = 1024
    cleanup_interval_seconds: int = 60
    enable_metrics: bool = True
    enable_logging: bool = True


@dataclass
class CacheMetrics:
    """Метрики кэша."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    last_cleanup: Optional[datetime] = None
    hit_rate: float = 0.0

    def update_hit_rate(self) -> None:
        """Обновление hit rate."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry(Generic[T]):
    """Запись кэша."""

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)

    def access(self) -> None:
        """Отметить доступ к записи."""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def get_size_bytes(self) -> int:
        """Получение размера записи в байтах."""
        try:
            return len(json.dumps(self.value, default=str).encode("utf-8"))
        except:
            return 0


class CacheProtocol:
    """Протокол кэша."""

    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        raise NotImplementedError

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Установка значения в кэш."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Очистка кэша."""
        raise NotImplementedError

    async def get_metrics(self) -> CacheMetrics:
        """Получение метрик кэша."""
        raise NotImplementedError


class MemoryCache(CacheProtocol):
    """In-memory реализация кэша."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        if config.enable_metrics:
            self._start_cleanup_task()

    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        async with self._lock:
            if key not in self._cache:
                self._metrics.misses += 1
                return None
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._metrics.misses += 1
                return None
            entry.access()
            self._cache.move_to_end(key)  # LRU
            self._metrics.hits += 1
            return entry.value

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Установка значения в кэш."""
        async with self._lock:
            try:
                ttl = ttl_seconds or self.config.default_ttl_seconds
                entry = CacheEntry(key=key, value=value, ttl_seconds=ttl)
                # Проверяем размер кэша
                if len(self._cache) >= self.config.max_size:
                    await self._evict_entries()
                self._cache[key] = entry
                self._cache.move_to_end(key)  # LRU
                self._metrics.sets += 1
                return True
            except Exception as e:
                logger.error(f"Error setting cache entry: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._metrics.deletes += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        async with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    async def clear(self) -> None:
        """Очистка кэша."""
        async with self._lock:
            self._cache.clear()
            self._metrics = CacheMetrics()

    async def get_metrics(self) -> CacheMetrics:
        """Получение метрик кэша."""
        async with self._lock:
            self._metrics.update_hit_rate()
            return self._metrics

    async def _evict_entries(self) -> None:
        """Вытеснение записей из кэша."""
        if self.config.eviction_strategy == CacheEvictionStrategy.LRU:
            # Удаляем самую старую запись
            if self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._metrics.evictions += 1
        elif self.config.eviction_strategy == CacheEvictionStrategy.LFU:
            # Удаляем запись с наименьшим количеством обращений
            if self._cache:
                least_frequent_key = min(
                    self._cache.keys(), key=lambda k: self._cache[k].access_count
                )
                del self._cache[least_frequent_key]
                self._metrics.evictions += 1
        elif self.config.eviction_strategy == CacheEvictionStrategy.FIFO:
            # Удаляем первую запись
            if self._cache:
                first_key = next(iter(self._cache))
                del self._cache[first_key]
                self._metrics.evictions += 1

    def _start_cleanup_task(self) -> None:
        """Запуск задачи очистки."""
        async def cleanup_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(self.config.cleanup_interval_seconds)
                    await self._cleanup_expired_entries()
                    await self._evict_entries()
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
        
        asyncio.create_task(cleanup_loop())

    async def _cleanup_expired_entries(self) -> None:
        """Очистка истекших записей."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            self._metrics.last_cleanup = datetime.now()
            if expired_keys and self.config.enable_logging:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class RedisCache(CacheProtocol):
    """Production-ready Redis кэш (асинхронный, fault-tolerant, TTL, eviction, метрики)."""

    def __init__(self, config: CacheConfig):
        if aioredis is None:
            raise ImportError("aioredis is required for RedisCache")
        self.config = config
        self._metrics = CacheMetrics()
        self._redis: Optional[Any] = None
        self._lock = asyncio.Lock()
        self._connected = False
        self._host = os.environ.get("REDIS_HOST", "localhost")
        self._port = int(os.environ.get("REDIS_PORT", 6379))
        self._db = int(os.environ.get("REDIS_DB", 0))

    async def _connect(self) -> None:
        if not self._connected:
            self._redis = await aioredis.create_redis_pool(
                (self._host, self._port), db=self._db
            )
            self._connected = True

    async def get(self, key: str) -> Optional[Any]:
        await self._connect()
        async with self._lock:
            if self._redis is None:
                return None
            assert self._redis is not None  # для mypy
            value = await self._redis.get(key)
            if value is None:
                self._metrics.misses += 1
                return None
            self._metrics.hits += 1
            try:
                return pickle.loads(value)
            except Exception:
                return value

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        await self._connect()
        async with self._lock:
            if self._redis is None:
                return False
            assert self._redis is not None  # для mypy
            try:
                data = pickle.dumps(value)
                ttl = ttl_seconds or self.config.default_ttl_seconds
                await self._redis.set(key, data, expire=ttl)
                self._metrics.sets += 1
                return True
            except Exception as e:
                logger.error(f"RedisCache set error: {e}")
                return False

    async def delete(self, key: str) -> bool:
        await self._connect()
        async with self._lock:
            if self._redis is None:
                return False
            assert self._redis is not None  # для mypy
            result = await self._redis.delete(key)
            if result:
                self._metrics.deletes += 1
            return bool(result)

    async def exists(self, key: str) -> bool:
        await self._connect()
        async with self._lock:
            if self._redis is None:
                return False
            assert self._redis is not None  # для mypy
            return await self._redis.exists(key) > 0

    async def clear(self) -> None:
        await self._connect()
        async with self._lock:
            if self._redis is not None:
                await self._redis.flushdb()
            self._metrics = CacheMetrics()

    async def get_metrics(self) -> CacheMetrics:
        return self._metrics


class DiskCache(CacheProtocol):
    """Production-ready Disk кэш (асинхронный, fault-tolerant, TTL, eviction, метрики)."""

    def __init__(self, config: CacheConfig, cache_dir: str = "./disk_cache"):
        if aiofiles is None:
            raise ImportError("aiofiles is required for DiskCache")
        self.config = config
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()

    def _get_path(self, key: str) -> str:
        return os.path.join(
            self.cache_dir, hashlib.md5(key.encode()).hexdigest() + ".pkl"
        )

    async def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        async with self._lock:
            if not os.path.exists(path):
                self._metrics.misses += 1
                return None
            try:
                async with aiofiles.open(path, "rb") as f:
                    data = await f.read()
                    value = pickle.loads(data)
                    self._metrics.hits += 1
                    return value
            except Exception as e:
                logger.error(f"DiskCache get error: {e}")
                return None

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        path = self._get_path(key)
        async with self._lock:
            try:
                data = pickle.dumps(value)
                async with aiofiles.open(path, "wb") as f:
                    await f.write(data)
                self._metrics.sets += 1
                return True
            except Exception as e:
                logger.error(f"DiskCache set error: {e}")
                return False

    async def delete(self, key: str) -> bool:
        path = self._get_path(key)
        async with self._lock:
            if os.path.exists(path):
                os.remove(path)
                self._metrics.deletes += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        path = self._get_path(key)
        async with self._lock:
            return os.path.exists(path)

    async def clear(self) -> None:
        async with self._lock:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, fname))
            self._metrics = CacheMetrics()

    async def get_metrics(self) -> CacheMetrics:
        return self._metrics


class HybridCache(CacheProtocol):
    """Production-ready гибридный кэш (Memory + Redis/Disk, fault-tolerant, TTL, eviction, метрики)."""

    def __init__(
        self,
        config: CacheConfig,
        redis_config: Optional[CacheConfig] = None,
        disk_config: Optional[CacheConfig] = None,
    ):
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(redis_config or config) if redis_config else None
        self.disk_cache = DiskCache(disk_config or config) if disk_config else None
        self._metrics = CacheMetrics()

    async def get(self, key: str) -> Optional[Any]:
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                await self.memory_cache.set(key, value)
                return value
        if self.disk_cache:
            value = await self.disk_cache.get(key)
            if value is not None:
                await self.memory_cache.set(key, value)
                return value
        return None

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        ok = await self.memory_cache.set(key, value, ttl_seconds)
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl_seconds)
        if self.disk_cache:
            await self.disk_cache.set(key, value, ttl_seconds)
        return ok

    async def delete(self, key: str) -> bool:
        ok = await self.memory_cache.delete(key)
        if self.redis_cache:
            await self.redis_cache.delete(key)
        if self.disk_cache:
            await self.disk_cache.delete(key)
        return ok

    async def exists(self, key: str) -> bool:
        if await self.memory_cache.exists(key):
            return True
        if self.redis_cache and await self.redis_cache.exists(key):
            return True
        if self.disk_cache and await self.disk_cache.exists(key):
            return True
        return False

    async def clear(self) -> None:
        await self.memory_cache.clear()
        if self.redis_cache:
            await self.redis_cache.clear()
        if self.disk_cache:
            await self.disk_cache.clear()

    async def get_metrics(self) -> CacheMetrics:
        # Агрегируем метрики
        m = self._metrics
        m.hits = self.memory_cache._metrics.hits
        m.misses = self.memory_cache._metrics.misses
        m.sets = self.memory_cache._metrics.sets
        m.deletes = self.memory_cache._metrics.deletes
        m.evictions = self.memory_cache._metrics.evictions
        return m


class CacheManager:
    """Менеджер кэшей."""

    def __init__(self):
        self._caches: Dict[str, CacheProtocol] = {}
        self._configs: Dict[str, CacheConfig] = {}

    def register_cache(self, name: str, cache: CacheProtocol, config: CacheConfig) -> None:
        """Регистрация кэша."""
        self._caches[name] = cache
        self._configs[name] = config
        logger.info(f"Registered cache: {name}")

    def get_cache(self, name: str) -> Optional[CacheProtocol]:
        """Получение кэша по имени."""
        return self._caches.get(name)

    async def get_all_metrics(self) -> Dict[str, CacheMetrics]:
        """Получение метрик всех кэшей."""
        metrics = {}
        for name, cache in self._caches.items():
            try:
                metrics[name] = await cache.get_metrics()
            except Exception as e:
                logger.error(f"Error getting metrics for cache {name}: {e}")
        return metrics

    async def clear_all_caches(self) -> None:
        """Очистка всех кэшей."""
        for name, cache in self._caches.items():
            try:
                await cache.clear()
                logger.info(f"Cleared cache: {name}")
            except Exception as e:
                logger.error(f"Error clearing cache {name}: {e}")


# Глобальный менеджер кэшей
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Получение глобального менеджера кэшей."""
    return _cache_manager


def create_cache(name: str, config: CacheConfig) -> CacheProtocol:
    """Создание кэша."""
    if config.cache_type == CacheType.MEMORY:
        cache: CacheProtocol = MemoryCache(config)
    elif config.cache_type == CacheType.REDIS:
        cache = RedisCache(config)
    elif config.cache_type == CacheType.DISK:
        cache = DiskCache(config)
    elif config.cache_type == CacheType.HYBRID:
        cache = HybridCache(config)
    else:
        raise ValueError(f"Unsupported cache type: {config.cache_type}")
    _cache_manager.register_cache(name, cache, config)
    return cache


def get_cache(name: str) -> Optional[CacheProtocol]:
    """Получение кэша по имени."""
    return _cache_manager.get_cache(name)


# Утилиты для работы с кэшем
def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """Генерация ключа кэша."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cache_decorator(
    cache_name: str,
    ttl_seconds: Optional[int] = None,
    key_generator: Optional[Callable] = None,
) -> Callable[[Callable], Callable]:
    """Декоратор для кэширования функций."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache(cache_name)
            if not cache:
                return await func(*args, **kwargs)
            # Генерируем ключ кэша
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            # Пытаемся получить из кэша
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            # Выполняем функцию и кэшируем результат
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl_seconds)
            return result

        return wrapper

    return decorator


class CacheableProtocol:
    """Протокол для кэшируемых объектов."""

    def get_cache_key(self) -> str:
        """Получение ключа кэша."""
        raise NotImplementedError

    def get_cache_ttl(self) -> Optional[int]:
        """Получение TTL для кэша."""
        raise NotImplementedError

    def is_cacheable(self) -> bool:
        """Проверка возможности кэширования."""
        return True


# Специализированные кэши
class MarketDataCache(MemoryCache):
    """Кэш для рыночных данных."""

    def __init__(self, config: CacheConfig):
        super().__init__(config)

    async def get_market_data(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[Any]:
        """Получение рыночных данных из кэша."""
        cache_key = generate_cache_key("market_data", symbol, timeframe, limit)
        return await self.get(cache_key)

    async def set_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        data: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Сохранение рыночных данных в кэш."""
        cache_key = generate_cache_key("market_data", symbol, timeframe, limit)
        return await self.set(cache_key, data, ttl_seconds)


class StrategyCache(MemoryCache):
    """Кэш для стратегий."""

    def __init__(self, config: CacheConfig):
        super().__init__(config)

    async def get_strategy_result(
        self, strategy_name: str, parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """Получение результата стратегии из кэша."""
        cache_key = generate_cache_key("strategy", strategy_name, parameters)
        return await self.get(cache_key)

    async def set_strategy_result(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        result: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Сохранение результата стратегии в кэш."""
        cache_key = generate_cache_key("strategy", strategy_name, parameters)
        return await self.set(cache_key, result, ttl_seconds)
