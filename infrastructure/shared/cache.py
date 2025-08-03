"""
Унифицированная система кэширования для Infrastructure Layer.
"""

import asyncio
import json
import pickle
import time
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import threading
from enum import Enum
import redis.asyncio as redis
from pathlib import Path

logger = logging.getLogger(__name__)

class CompressionType(Enum):
    """Типы сжатия данных."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"

class EncryptionType(Enum):
    """Типы шифрования данных."""
    NONE = "none"
    AES = "aes"
    FERNET = "fernet"

@dataclass
class CacheConfig:
    """Конфигурация кэша."""
    max_size: int = 10000
    default_ttl: int = 3600
    cleanup_interval: int = 300
    compression: CompressionType = CompressionType.NONE
    encryption: EncryptionType = EncryptionType.NONE
    persistence_path: Optional[str] = None
    redis_url: Optional[str] = None
    enable_metrics: bool = True

@dataclass
class CacheMetrics:
    """Метрики кэша."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    last_cleanup: Optional[datetime] = None
    
    @property
    def hit_ratio(self) -> float:
        """Коэффициент попаданий."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'size': self.size,
            'memory_usage': self.memory_usage,
            'hit_ratio': self.hit_ratio,
            'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None
        }

@dataclass
class CacheEntry:
    """Запись в кэше."""
    value: Any
    created_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size: int = 0
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Проверка истечения времени жизни."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Обновление времени доступа."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def add_tag(self, tag: str) -> None:
        """Добавление тега."""
        self.tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Проверка наличия тега."""
        return tag in self.tags

class CacheProtocol(ABC):
    """Протокол кэша."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Установка значения в кэш."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Очистка кэша."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> CacheMetrics:
        """Получение метрик кэша."""
        pass

class DataSerializer:
    """Сериализатор данных с поддержкой сжатия и шифрования."""
    
    def __init__(self, compression: CompressionType = CompressionType.NONE, 
                 encryption: EncryptionType = EncryptionType.NONE,
                 encryption_key: Optional[bytes] = None):
        self.compression = compression
        self.encryption = encryption
        self.encryption_key = encryption_key
        
        # Инициализация модулей сжатия
        self._compression_modules = {}
        if compression == CompressionType.GZIP:
            import gzip
            self._compression_modules['compress'] = gzip.compress
            self._compression_modules['decompress'] = gzip.decompress
        elif compression == CompressionType.ZLIB:
            import zlib
            self._compression_modules['compress'] = zlib.compress
            self._compression_modules['decompress'] = zlib.decompress
        elif compression == CompressionType.LZ4:
            try:
                import lz4.frame
                self._compression_modules['compress'] = lz4.frame.compress
                self._compression_modules['decompress'] = lz4.frame.decompress
            except ImportError:
                logger.warning("LZ4 не установлен, используется без сжатия")
                self.compression = CompressionType.NONE
        
        # Инициализация шифрования
        if encryption == EncryptionType.FERNET:
            try:
                from cryptography.fernet import Fernet
                if encryption_key:
                    self._fernet = Fernet(encryption_key)
                else:
                    self._fernet = Fernet(Fernet.generate_key())
            except ImportError:
                logger.warning("cryptography не установлена, шифрование отключено")
                self.encryption = EncryptionType.NONE
    
    def serialize(self, value: Any) -> bytes:
        """Сериализация значения."""
        try:
            # Сериализация в pickle
            data = pickle.dumps(value)
            
            # Сжатие
            if self.compression != CompressionType.NONE and 'compress' in self._compression_modules:
                data = self._compression_modules['compress'](data)
            
            # Шифрование
            if self.encryption == EncryptionType.FERNET and hasattr(self, '_fernet'):
                data = self._fernet.encrypt(data)
            
            return data
        except Exception as e:
            logger.error(f"Ошибка сериализации: {e}")
            raise
    
    def deserialize(self, data: bytes) -> Any:
        """Десериализация значения."""
        try:
            # Расшифровка
            if self.encryption == EncryptionType.FERNET and hasattr(self, '_fernet'):
                data = self._fernet.decrypt(data)
            
            # Распаковка
            if self.compression != CompressionType.NONE and 'decompress' in self._compression_modules:
                data = self._compression_modules['decompress'](data)
            
            # Десериализация
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Ошибка десериализации: {e}")
            raise

class MemoryCache(CacheProtocol):
    """Продвинутый in-memory кэш."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._storage: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._metrics = CacheMetrics()
        self._serializer = DataSerializer(
            compression=config.compression,
            encryption=config.encryption
        )
        
        # Индексы для быстрого поиска
        self._tags_index: Dict[str, Set[str]] = {}
        self._access_order: List[str] = []
        
        # Фоновые задачи
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Запуск фоновых задач."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        async with self._lock:
            entry = self._storage.get(key)
            if entry is None:
                self._metrics.misses += 1
                return None
            
            if entry.is_expired():
                await self._remove_entry(key)
                self._metrics.misses += 1
                self._metrics.evictions += 1
                return None
            
            entry.touch()
            self._update_access_order(key)
            self._metrics.hits += 1
            return entry.value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Установка значения в кэш."""
        try:
            async with self._lock:
                # Проверка лимита размера
                if len(self._storage) >= self.config.max_size and key not in self._storage:
                    await self._evict_lru_entries()
                
                # Сериализация для вычисления размера
                serialized_data = self._serializer.serialize(value)
                
                ttl = ttl_seconds or self.config.default_ttl
                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    ttl=ttl,
                    size=len(serialized_data),
                    tags=tags or set()
                )
                
                # Удаление старой записи если существует
                if key in self._storage:
                    await self._remove_entry(key)
                
                self._storage[key] = entry
                self._update_access_order(key)
                self._update_tags_index(key, entry.tags)
                
                self._metrics.sets += 1
                self._metrics.size = len(self._storage)
                self._metrics.memory_usage += entry.size
                
                return True
        except Exception as e:
            logger.error(f"Ошибка установки значения в кэш: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        async with self._lock:
            if key in self._storage:
                await self._remove_entry(key)
                self._metrics.deletes += 1
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        async with self._lock:
            entry = self._storage.get(key)
            if entry is None:
                return False
            
            if entry.is_expired():
                await self._remove_entry(key)
                self._metrics.evictions += 1
                return False
            
            return True
    
    async def clear(self) -> None:
        """Очистка кэша."""
        async with self._lock:
            self._storage.clear()
            self._tags_index.clear()
            self._access_order.clear()
            self._metrics = CacheMetrics()
    
    async def get_metrics(self) -> CacheMetrics:
        """Получение метрик кэша."""
        async with self._lock:
            self._metrics.size = len(self._storage)
            return self._metrics
    
    async def delete_by_tag(self, tag: str) -> int:
        """Удаление записей по тегу."""
        async with self._lock:
            keys_to_delete = self._tags_index.get(tag, set()).copy()
            deleted_count = 0
            
            for key in keys_to_delete:
                if key in self._storage:
                    await self._remove_entry(key)
                    deleted_count += 1
            
            return deleted_count
    
    async def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """Получение записей по тегу."""
        async with self._lock:
            result = {}
            keys = self._tags_index.get(tag, set())
            
            for key in keys:
                if key in self._storage:
                    entry = self._storage[key]
                    if not entry.is_expired():
                        result[key] = entry.value
                        entry.touch()
            
            return result
    
    async def _remove_entry(self, key: str) -> None:
        """Удаление записи и обновление индексов."""
        if key not in self._storage:
            return
        
        entry = self._storage[key]
        
        # Обновление индекса тегов
        for tag in entry.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(key)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
        
        # Обновление порядка доступа
        if key in self._access_order:
            self._access_order.remove(key)
        
        # Обновление метрик
        self._metrics.memory_usage -= entry.size
        
        del self._storage[key]
    
    def _update_access_order(self, key: str) -> None:
        """Обновление порядка доступа для LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _update_tags_index(self, key: str, tags: Set[str]) -> None:
        """Обновление индекса тегов."""
        for tag in tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(key)
    
    async def _evict_lru_entries(self) -> None:
        """Выселение наименее используемых записей."""
        if not self._access_order:
            return
        
        # Удаляем 10% наименее используемых записей
        evict_count = max(1, len(self._access_order) // 10)
        
        for _ in range(evict_count):
            if self._access_order:
                key = self._access_order[0]
                await self._remove_entry(key)
                self._metrics.evictions += 1
    
    async def _cleanup_loop(self) -> None:
        """Фоновая очистка истёкших записей."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в процессе очистки кэша: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Очистка истёкших записей."""
        async with self._lock:
            expired_keys = []
            
            for key, entry in self._storage.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._remove_entry(key)
                self._metrics.evictions += 1
            
            self._metrics.last_cleanup = datetime.now()
            
            if expired_keys:
                logger.debug(f"Удалено {len(expired_keys)} истёкших записей")

class RedisCache(CacheProtocol):
    """Кэш на основе Redis."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._redis: Optional[redis.Redis] = None
        self._metrics = CacheMetrics()
        self._serializer = DataSerializer(
            compression=config.compression,
            encryption=config.encryption
        )
    
    async def _get_redis(self) -> redis.Redis:
        """Получение подключения к Redis."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.config.redis_url or "redis://localhost:6379",
                decode_responses=False  # Работаем с bytes
            )
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        try:
            r = await self._get_redis()
            data = await r.get(key)
            
            if data is None:
                self._metrics.misses += 1
                return None
            
            value = self._serializer.deserialize(data)
            self._metrics.hits += 1
            return value
        except Exception as e:
            logger.error(f"Ошибка получения из Redis: {e}")
            self._metrics.misses += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Установка значения в кэш."""
        try:
            r = await self._get_redis()
            data = self._serializer.serialize(value)
            ttl = ttl_seconds or self.config.default_ttl
            
            result = await r.setex(key, ttl, data)
            if result:
                self._metrics.sets += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка установки в Redis: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        try:
            r = await self._get_redis()
            result = await r.delete(key)
            if result:
                self._metrics.deletes += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка удаления из Redis: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        try:
            r = await self._get_redis()
            return bool(await r.exists(key))
        except Exception as e:
            logger.error(f"Ошибка проверки существования в Redis: {e}")
            return False
    
    async def clear(self) -> None:
        """Очистка кэша."""
        try:
            r = await self._get_redis()
            await r.flushdb()
            self._metrics = CacheMetrics()
        except Exception as e:
            logger.error(f"Ошибка очистки Redis: {e}")
    
    async def get_metrics(self) -> CacheMetrics:
        """Получение метрик кэша."""
        try:
            r = await self._get_redis()
            info = await r.info('memory')
            self._metrics.memory_usage = info.get('used_memory', 0)
            
            db_info = await r.info('keyspace')
            db0_info = db_info.get('db0', {})
            if isinstance(db0_info, dict):
                self._metrics.size = db0_info.get('keys', 0)
            
            return self._metrics
        except Exception as e:
            logger.error(f"Ошибка получения метрик Redis: {e}")
            return self._metrics

class HybridCache(CacheProtocol):
    """Гибридный кэш с двумя уровнями."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._l1_cache = MemoryCache(config)
        self._l2_cache = RedisCache(config) if config.redis_url else None
        self._metrics = CacheMetrics()
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из двухуровневого кэша."""
        # Сначала проверяем L1 (память)
        value = await self._l1_cache.get(key)
        if value is not None:
            self._metrics.hits += 1
            return value
        
        # Затем проверяем L2 (Redis)
        if self._l2_cache:
            value = await self._l2_cache.get(key)
            if value is not None:
                # Записываем в L1 для быстрого доступа
                await self._l1_cache.set(key, value)
                self._metrics.hits += 1
                return value
        
        self._metrics.misses += 1
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Установка значения в двухуровневый кэш."""
        success = True
        
        # Записываем в L1
        l1_result = await self._l1_cache.set(key, value, ttl_seconds)
        success = success and l1_result
        
        # Записываем в L2
        if self._l2_cache:
            l2_result = await self._l2_cache.set(key, value, ttl_seconds)
            success = success and l2_result
        
        if success:
            self._metrics.sets += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Удаление значения из двухуровневого кэша."""
        l1_result = await self._l1_cache.delete(key)
        l2_result = True
        
        if self._l2_cache:
            l2_result = await self._l2_cache.delete(key)
        
        if l1_result or l2_result:
            self._metrics.deletes += 1
            return True
        
        return False
    
    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        if await self._l1_cache.exists(key):
            return True
        
        if self._l2_cache:
            return await self._l2_cache.exists(key)
        
        return False
    
    async def clear(self) -> None:
        """Очистка двухуровневого кэша."""
        await self._l1_cache.clear()
        if self._l2_cache:
            await self._l2_cache.clear()
        self._metrics = CacheMetrics()
    
    async def get_metrics(self) -> CacheMetrics:
        """Получение объединённых метрик."""
        l1_metrics = await self._l1_cache.get_metrics()
        l2_metrics = None
        
        if self._l2_cache:
            l2_metrics = await self._l2_cache.get_metrics()
        
        # Объединяем метрики
        combined_metrics = CacheMetrics(
            hits=self._metrics.hits,
            misses=self._metrics.misses,
            sets=self._metrics.sets,
            deletes=self._metrics.deletes,
            evictions=l1_metrics.evictions + (l2_metrics.evictions if l2_metrics else 0),
            size=l1_metrics.size + (l2_metrics.size if l2_metrics else 0),
            memory_usage=l1_metrics.memory_usage + (l2_metrics.memory_usage if l2_metrics else 0)
        )
        
        return combined_metrics

class CacheFactory:
    """Фабрика для создания кэшей."""
    
    @staticmethod
    def create_cache(config: CacheConfig) -> CacheProtocol:
        """Создание кэша согласно конфигурации."""
        if config.redis_url:
            if config.max_size > 0:
                # Гибридный кэш
                return HybridCache(config)
            else:
                # Только Redis
                return RedisCache(config)
        else:
            # Только память
            return MemoryCache(config)

class CacheManager:
    """Менеджер кэшей с поддержкой множественных экземпляров."""
    
    def __init__(self):
        self._caches: Dict[str, CacheProtocol] = {}
        self._default_config = CacheConfig()
    
    def register_cache(
        self, 
        name: str, 
        config: Optional[CacheConfig] = None
    ) -> CacheProtocol:
        """Регистрация нового кэша."""
        cache_config = config or self._default_config
        cache = CacheFactory.create_cache(cache_config)
        self._caches[name] = cache
        return cache
    
    def get_cache(self, name: str = "default") -> CacheProtocol:
        """Получение кэша по имени."""
        if name not in self._caches:
            self._caches[name] = CacheFactory.create_cache(self._default_config)
        return self._caches[name]
    
    async def get_all_metrics(self) -> Dict[str, CacheMetrics]:
        """Получение метрик всех кэшей."""
        metrics = {}
        for name, cache in self._caches.items():
            metrics[name] = await cache.get_metrics()
        return metrics
    
    async def clear_all(self) -> None:
        """Очистка всех кэшей."""
        for cache in self._caches.values():
            await cache.clear()

# Глобальный менеджер кэшей
cache_manager = CacheManager()

# Регистрация кэша по умолчанию
default_cache_config = CacheConfig(
    max_size=10000,
    default_ttl=3600,
    compression=CompressionType.GZIP,
    enable_metrics=True
)
cache_manager.register_cache("default", default_cache_config)

def cache_key(*args, **kwargs) -> str:
    """Генерация ключа кэша из аргументов."""
    key_parts = []
    
    # Добавляем позиционные аргументы
    for arg in args:
        if hasattr(arg, '__dict__'):
            key_parts.append(str(hash(str(arg.__dict__))))
        else:
            key_parts.append(str(hash(str(arg))))
    
    # Добавляем именованные аргументы
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{hash(str(v))}")
    
    # Создаём хеш от всех частей
    combined = ":".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()

def cached(
    ttl: Optional[int] = None,
    cache_name: str = "default",
    key_func: Optional[callable] = None,
    tags: Optional[Set[str]] = None
):
    """Декоратор для кэширования результатов функций."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name)
            
            # Генерация ключа
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = f"{func.__module__}.{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Попытка получить из кэша
            cached_result = await cache.get(cache_key_str)
            if cached_result is not None:
                return cached_result
            
            # Выполнение функции
            result = await func(*args, **kwargs)
            
            # Кэширование результата
            await cache.set(cache_key_str, result, ttl)
            
            return result
        
        return wrapper
    return decorator
