"""
Промышленный сервис кэширования для application слоя.
Реализует протокол CacheServiceProtocol из application/protocols.py
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from application.types import CacheConfig


class CacheError(Exception):
    """Ошибка сервиса кэширования."""

    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CacheStrategy(str, Enum):
    """Стратегии кэширования."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """Запись в кэше."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # Time to live в секундах
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)

    def update_access(self) -> None:
        """Обновление времени доступа."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Статистика кэша."""

    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def hit_rate(self) -> float:
        """Коэффициент попаданий."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Коэффициент промахов."""
        return 1.0 - self.hit_rate


class CacheStorageProtocol(ABC):
    """Протокол для хранилища кэша."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Получение записи."""
        ...

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Установка записи."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Удаление записи."""
        ...

    @abstractmethod
    async def clear(self) -> bool:
        """Очистка кэша."""
        ...

    @abstractmethod
    async def get_all_keys(self) -> List[str]:
        """Получение всех ключей."""
        ...

    @abstractmethod
    async def get_size(self) -> int:
        """Получение размера кэша."""
        ...


class InMemoryCacheStorage(CacheStorageProtocol):
    """Хранилище кэша в памяти."""

    def __init__(self, *args, **kwargs) -> Any:
        self.max_size = max_size
        self.storage: Dict[str, CacheEntry] = {}
        self.logger = logging.getLogger(__name__)

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Получение записи."""
        entry = self.storage.get(key)
        if entry and not entry.is_expired():
            entry.update_access()
            return entry
        elif entry and entry.is_expired():
            await self.delete(key)
        return None  # type: None

    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Установка записи."""
        try:
            # Проверяем размер кэша
            if len(self.storage) >= self.max_size:
                await self._evict_entries()
            self.storage[key] = entry
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache entry: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление записи."""
        try:
            if key in self.storage:
                del self.storage[key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry: {e}")
            return False

    async def clear(self) -> bool:
        """Очистка кэша."""
        try:
            self.storage.clear()
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_all_keys(self) -> List[str]:
        """Получение всех ключей."""
        return list(self.storage.keys())

    async def get_size(self) -> int:
        """Получение размера кэша."""
        return len(self.storage)

    async def _evict_entries(self) -> None:
        """Вытеснение записей по стратегии LRU."""
        if not self.storage:
            return
        # Удаляем истекшие записи
        expired_keys = [
            key for key, entry in self.storage.items() if entry.is_expired()
        ]
        for key in expired_keys:
            await self.delete(key)
        # Если все еще превышаем лимит, удаляем самые старые
        if len(self.storage) >= self.max_size:
            sorted_entries = sorted(
                self.storage.items(), key=lambda x: x[1].accessed_at
            )
            # Удаляем 10% самых старых записей
            to_remove = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:to_remove]:
                await self.delete(key)


class CacheService:
    """Промышленный сервис кэширования."""

    def __init__(self, *args, **kwargs) -> Any:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.storage = InMemoryCacheStorage(config.max_size)
        self.stats = CacheStats()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        # Регистрируем сериализаторы для разных типов данных
        self.serializers: Dict[str, Callable] = {
            "json": self._serialize_json,
            "pickle": self._serialize_pickle,
            "string": self._serialize_string,
        }

    async def start(self) -> Any:
        """Запуск сервиса кэширования."""
        if self.is_running:
            return
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Cache service started")

    async def stop(self) -> Any:
        """Остановка сервиса кэширования."""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cache service stopped")

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Получение записи с AI-оптимизированным предварительным кэшированием."""
        try:
            entry = self.storage.get(key)
            
            if entry and not entry.is_expired():
                entry.update_access()
                
                # AI-предсказание следующих запросов
                await self._predict_and_prefetch(key)
                
                # Обновление статистики использования
                await self._update_access_patterns(key)
                
                return entry
            elif entry and entry.is_expired():
                await self.delete(key)
                
                # Попытка автоматического обновления кэша
                refreshed_entry = await self._attempt_auto_refresh(key)
                if refreshed_entry:
                    return refreshed_entry
            
            # Запись не найдена - анализируем паттерны для предложений
            await self._analyze_miss_patterns(key)
            
            return None  # type: None
            
        except Exception as e:
            self.logger.error(f"Ошибка получения записи из кэша {key}: {e}")
            return None  # type: None
    
    async def _predict_and_prefetch(self, accessed_key: str) -> None:
        """AI-предсказание и предварительная загрузка связанных данных."""
        try:
            # Анализ паттернов доступа для предсказания следующих запросов
            related_keys = await self._predict_related_keys(accessed_key)
            
            # Предварительная загрузка связанных данных
            for related_key in related_keys:
                if related_key not in self.storage:
                    await self._prefetch_data(related_key)
                    
        except Exception as e:
            self.logger.debug(f"Ошибка предсказания кэша для {accessed_key}: {e}")
    
    async def _predict_related_keys(self, key: str) -> List[str]:
        """AI-предсказание связанных ключей на основе паттернов."""
        try:
            # Анализ исторических паттернов доступа
            if hasattr(self, 'access_patterns'):
                pattern = self.access_patterns.get(key, {})
                related = pattern.get('often_accessed_with', [])
                
                # Машинное обучение для улучшения предсказаний
                predicted = await self._ml_predict_related_keys(key, related)
                return predicted[:5]  # Ограничиваем количество предсказаний
            
            return []  # type: List[Any]
        except Exception:
            return []  # type: List[Any]
    
    async def _ml_predict_related_keys(self, key: str, historical_related: List[str]) -> List[str]:
        """Машинное обучение для предсказания связанных ключей."""
        try:
            # Простая эвристика на основе схожести ключей
            predictions = []
            
            for existing_key in self.storage.keys():
                if existing_key != key:
                    similarity = await self._calculate_key_similarity(key, existing_key)
                    if similarity > 0.7:  # Высокая схожесть
                        predictions.append(existing_key)
            
            # Комбинируем с историческими данными
            combined = list(set(historical_related + predictions))
            return combined
            
        except Exception:
            return historical_related
    
    async def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """Расчет семантической схожести ключей."""
        try:
            # Простая метрика на основе общих подстрок
            key1_parts = set(key1.split('_'))
            key2_parts = set(key2.split('_'))
            
            if not key1_parts or not key2_parts:
                return 0.0
            
            intersection = len(key1_parts.intersection(key2_parts))
            union = len(key1_parts.union(key2_parts))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _prefetch_data(self, key: str) -> None:
        """Предварительная загрузка данных в кэш."""
        try:
            # Здесь можно реализовать логику загрузки данных из источника
            # Для демонстрации создаем заглушку
            if hasattr(self, 'data_loader'):
                data = await self.data_loader.load(key)
                if data:
                    entry = CacheEntry(
                        key=key,
                        value=data,
                        ttl=300,  # 5 минут
                        metadata={'prefetched': True}
                    )
                    await self.set(key, entry)
                    
        except Exception as e:
            self.logger.debug(f"Ошибка предварительной загрузки {key}: {e}")
    
    async def _update_access_patterns(self, key: str) -> None:
        """Обновление паттернов доступа для ML."""
        try:
            if not hasattr(self, 'access_patterns'):
                self.access_patterns = {}
            
            pattern = self.access_patterns.get(key, {
                'access_count': 0,
                'last_accessed': [],
                'often_accessed_with': [],
                'access_times': []
            })
            
            pattern['access_count'] += 1
            pattern['access_times'].append(datetime.now())
            
            # Ограничиваем историю
            if len(pattern['access_times']) > 100:
                pattern['access_times'] = pattern['access_times'][-50:]
            
            self.access_patterns[key] = pattern
            
        except Exception as e:
            self.logger.debug(f"Ошибка обновления паттернов для {key}: {e}")
    
    async def _attempt_auto_refresh(self, key: str) -> Optional[CacheEntry]:
        """Попытка автоматического обновления устаревшего кэша."""
        try:
            # Проверяем, есть ли загрузчик данных
            if hasattr(self, 'data_loader'):
                refreshed_data = await self.data_loader.load(key)
                if refreshed_data:
                    entry = CacheEntry(
                        key=key,
                        value=refreshed_data,
                        ttl=300,  # 5 минут
                        metadata={'auto_refreshed': True}
                    )
                    await self.set(key, entry)
                    return entry
            
            return None  # type: None
            
        except Exception as e:
            self.logger.debug(f"Ошибка автообновления кэша для {key}: {e}")
            return None  # type: None
    
    async def _analyze_miss_patterns(self, key: str) -> None:
        """Анализ паттернов промахов кэша для оптимизации."""
        try:
            if not hasattr(self, 'miss_patterns'):
                self.miss_patterns = {}
            
            pattern = self.miss_patterns.get(key, {
                'miss_count': 0,
                'miss_times': []
            })
            
            pattern['miss_count'] += 1
            pattern['miss_times'].append(datetime.now())
            
            # Ограничиваем историю
            if len(pattern['miss_times']) > 50:
                pattern['miss_times'] = pattern['miss_times'][-25:]
            
            self.miss_patterns[key] = pattern
            
            # Если слишком много промахов, предлагаем предварительную загрузку
            if pattern['miss_count'] > 5:
                await self._schedule_proactive_loading(key)
                
        except Exception as e:
            self.logger.debug(f"Ошибка анализа промахов для {key}: {e}")
    
    async def _schedule_proactive_loading(self, key: str) -> None:
        """Планирование проактивной загрузки часто запрашиваемых данных."""
        try:
            # Добавляем в очередь для проактивной загрузки
            if not hasattr(self, 'proactive_queue'):
                self.proactive_queue = set()
            
            self.proactive_queue.add(key)
            
            # Запускаем фоновую задачу для загрузки
            asyncio.create_task(self._background_proactive_load(key))
            
        except Exception as e:
            self.logger.debug(f"Ошибка планирования проактивной загрузки для {key}: {e}")
    
    async def _background_proactive_load(self, key: str) -> None:
        """Фоновая проактивная загрузка данных."""
        try:
            await asyncio.sleep(1)  # Небольшая задержка
            
            if key in getattr(self, 'proactive_queue', set()):
                await self._prefetch_data(key)
                
                if hasattr(self, 'proactive_queue'):
                    self.proactive_queue.discard(key)
                    
        except Exception as e:
            self.logger.debug(f"Ошибка фоновой загрузки для {key}: {e}")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
        try:
            # Создаем запись кэша
            entry = CacheEntry(key=key, value=value, ttl=ttl or self.config.ttl_seconds)
            success = await self.storage.set(key, entry)
            if success:
                self.stats.total_entries = await self.storage.get_size()
            return success
        except Exception as e:
            self.logger.error(f"Failed to set cache value: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        try:
            success = await self.storage.delete(key)
            if success:
                self.stats.total_entries = await self.storage.get_size()
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete from cache: {e}")
            return False

    async def clear(self, pattern: str = "*") -> bool:
        """Очистка кэша по паттерну."""
        try:
            if pattern == "*":
                success = await self.storage.clear()
                if success:
                    self.stats.total_entries = 0
                return success
            else:
                # Удаляем по паттерну
                keys = await self.storage.get_all_keys()
                deleted_count = 0
                for key in keys:
                    if self._matches_pattern(key, pattern):
                        if await self.storage.delete(key):
                            deleted_count += 1
                self.stats.total_entries = await self.storage.get_size()
                return deleted_count > 0
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        try:
            return {
                "total_entries": self.stats.total_entries,
                "hit_count": self.stats.hit_count,
                "miss_count": self.stats.miss_count,
                "hit_rate": self.stats.hit_rate,
                "miss_rate": self.stats.miss_rate,
                "eviction_count": self.stats.eviction_count,
                "total_size_bytes": self.stats.total_size_bytes,
                "created_at": self.stats.created_at.isoformat(),
                "uptime_seconds": (
                    datetime.now() - self.stats.created_at
                ).total_seconds(),
                "config": {
                    "max_size": self.config.max_size,
                    "ttl_seconds": self.config.ttl_seconds,
                    "cleanup_interval": self.config.cleanup_interval,
                },
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}  # type: Dict[str, Any]

    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Получение нескольких значений."""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_multi(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Установка нескольких значений."""
        try:
            success_count = 0
            for key, value in data.items():
                if await self.set(key, value, ttl):
                    success_count += 1
            return success_count == len(data)
        except Exception as e:
            self.logger.error(f"Failed to set multiple cache values: {e}")
            return False

    async def delete_multi(self, keys: List[str]) -> bool:
        """Удаление нескольких значений."""
        try:
            success_count = 0
            for key in keys:
                if await self.delete(key):
                    success_count += 1
            return success_count == len(keys)
        except Exception as e:
            self.logger.error(f"Failed to delete multiple cache values: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        try:
            entry = await self.storage.get(key)
            return entry is not None
        except Exception as e:
            self.logger.error(f"Failed to check cache key existence: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Увеличение числового значения."""
        try:
            current_value = await self.get(key)
            if current_value is None:
                new_value = amount
            elif isinstance(current_value, (int, float)):
                new_value = int(current_value) + amount
            else:
                raise CacheError(
                    f"Cannot increment non-numeric value: {type(current_value)}"
                )
            await self.set(key, new_value)
            return new_value
        except Exception as e:
            self.logger.error(f"Failed to increment cache value: {e}")
            return None  # type: None

    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Уменьшение числового значения."""
        return await self.increment(key, -amount)

    def generate_key(self, *args, **kwargs) -> str:
        """Генерация ключа кэша из аргументов."""
        try:
            # Создаем строку из аргументов
            key_parts = []
            for arg in args:
                key_parts.append(str(arg))
            for key, value in sorted(kwargs.items()):
                key_parts.append(f"{key}:{value}")
            key_string = "|".join(key_parts)
            # Создаем хеш
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to generate cache key: {e}")
            return str(hash(str(args) + str(kwargs)))

    async def _cleanup_loop(self) -> Any:
        """Цикл очистки истекших записей."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {e}")

    async def _cleanup_expired_entries(self) -> Any:
        """Очистка истекших записей."""
        try:
            keys = await self.storage.get_all_keys()
            expired_count = 0
            for key in keys:
                entry = await self.storage.get(key)
                if entry and entry.is_expired():
                    if await self.storage.delete(key):
                        expired_count += 1
            if expired_count > 0:
                self.stats.eviction_count += expired_count
                self.stats.total_entries = await self.storage.get_size()
                self.logger.debug(f"Cleaned up {expired_count} expired cache entries")
        except Exception as e:
            self.logger.error(f"Error cleaning up expired cache entries: {e}")

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Проверка соответствия ключа паттерну."""
        if pattern == "*":
            return True
        # Простая реализация паттернов
        if "*" in pattern:
            # Заменяем * на .* для regex
            import re

            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(regex_pattern, key))
        return key == pattern

    def _serialize_json(self, value: Any) -> str:
        """Сериализация в JSON."""
        return json.dumps(value, default=str)

    def _serialize_pickle(self, value: Any) -> bytes:
        """Сериализация через pickle."""
        import pickle

        return pickle.dumps(value)

    def _serialize_string(self, value: Any) -> str:
        """Сериализация в строку."""
        return str(value)

    def get_serialized_size(self, value: Any) -> int:
        """Получение размера сериализованного значения."""
        try:
            serialized = self._serialize_json(value)
            return len(serialized.encode("utf-8"))
        except Exception:
            return 0
