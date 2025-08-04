"""
Промышленные базовые интерфейсы репозиториев.
Определяет общие интерфейсы для всех репозиториев в системе,
следуя принципам DDD и обеспечивая единообразие API.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
    Callable,
    ContextManager,
    Deque,
)
from uuid import UUID
from collections import deque

from domain.exceptions.protocol_exceptions import (
    ConnectionError,
    EntitySaveError,
    EntityUpdateError,
    TimeoutError,
    TransactionError,
)
from domain.types.repository_types import (
    BulkOperationResult,
    CacheEvictionStrategy,
    CacheKey,
    CacheMetrics,
    CacheTTL,
    CacheType,
    EntityId,
    HealthCheckResult,
    Pagination,
    PerformanceMetrics,
    QueryFilter,
    QueryOperator,
    QueryOptions,
    RepositoryConfig,
    RepositoryMetrics,
    RepositoryResponse,
    RepositoryState,
    SortOrder,
    TransactionId,
    TransactionMetrics,
    TransactionStatus,
)

T = TypeVar("T")


@runtime_checkable
class TransactionProtocol(Protocol):
    """Протокол транзакции."""

    async def __aenter__(self) -> "TransactionProtocol": ...
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def is_active(self) -> bool: ...
@runtime_checkable
class ConnectionProtocol(Protocol):
    """Протокол соединения с БД."""

    async def get_transaction(self) -> TransactionProtocol: ...
    async def close(self) -> None: ...
    async def is_connected(self) -> bool: ...
    async def ping(self) -> bool: ...
    async def get_connection_info(self) -> Dict[str, Any]: ...
class BaseRepository(ABC, Generic[T]):
    """
    Промышленный базовый репозиторий с полной реализацией.
    Обеспечивает CRUD операции с типизированными доменными сущностями:
    - Создание и сохранение с валидацией
    - Чтение и поиск с фильтрацией и пагинацией
    - Обновление с оптимистичной блокировкой
    - Удаление с мягким удалением
    - Транзакционность и кэширование
    - Пакетные операции и мониторинг
    """

    def __init__(self, config: Optional[RepositoryConfig] = None):
        """Инициализация репозитория."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or RepositoryConfig()
        # Кэширование
        self._cache: Dict[CacheKey, T] = {}
        self._cache_ttl: Dict[CacheKey, datetime] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._cache_max_size = self.config.get("cache_max_size", 1000)
        self._cache_ttl_seconds = self.config.get("cache_ttl", 300)
        self._cache_type = CacheType.MEMORY
        self._eviction_strategy = CacheEvictionStrategy.LRU
        # Состояние
        self._state = RepositoryState.INITIALIZING
        # Метрики и статистика
        self._operation_count = 0
        self._query_times: Deque[float] = deque(maxlen=1000)
        self._error_count = 0  # Добавляем недостающий атрибут
        self._last_error: Optional[str] = None
        self._startup_time = datetime.now()
        self._last_cleanup = datetime.now()
        # Метрики
        self._metrics = RepositoryMetrics(
            total_entities=0,
            cache_hit_rate=0.0,
            avg_query_time=0.0,
            error_rate=0.0,
            last_cleanup=self._last_cleanup.isoformat(),
            uptime_seconds=0.0,
            memory_usage_mb=0.0,
            disk_usage_mb=0.0,
        )
        # Транзакции
        self._transactions: Dict[TransactionId, Dict[str, Any]] = {}
        self._transaction_locks: Dict[str, asyncio.Lock] = {}
        self._transaction_metrics = TransactionMetrics(
            total_transactions=0,
            committed_transactions=0,
            rolled_back_transactions=0,
            avg_transaction_time=0.0,
            concurrent_transactions=0,
            deadlocks=0,
        )
        # Производительность
        # Инициализация
        self._initialize_repository()
        self._state = RepositoryState.CONNECTED
        self.logger.info(f"{self.__class__.__name__} initialized successfully")

    def _initialize_repository(self) -> None:
        """Инициализация репозитория."""
        try:
            # Запуск фоновых задач
            if self.config.get("enable_metrics", True):
                asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._background_cleanup())
            self.logger.debug("Repository initialization completed")
        except Exception as e:
            self.logger.error(f"Repository initialization failed: {e}")
            self._state = RepositoryState.ERROR
            raise

    @property
    def state(self) -> RepositoryState:
        """Текущее состояние репозитория."""
        return self._state

    @property
    def is_healthy(self) -> bool:
        """Проверка здоровья репозитория."""
        return self._state == RepositoryState.CONNECTED

    # ============================================================================
    # БАЗОВЫЕ CRUD ОПЕРАЦИИ
    # ============================================================================
    @abstractmethod
    async def save(self, entity: T) -> T:
        """
        Сохранение сущности.
        Args:
            entity: Сущность для сохранения
        Returns:
            T: Сохраненная сущность
        Raises:
            EntitySaveError: Ошибка сохранения
            ValidationError: Ошибка валидации
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: EntityId) -> Optional[T]:
        """
        Получение сущности по ID.
        Args:
            entity_id: ID сущности
        Returns:
            Optional[T]: Сущность или None
        Raises:
            EntityNotFoundError: Сущность не найдена
        """
        pass

    @abstractmethod
    async def get_all(self, options: Optional[QueryOptions] = None) -> List[T]:
        """
        Получение всех сущностей.
        Args:
            options: Опции запроса
        Returns:
            List[T]: Список сущностей
        """
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Обновление сущности.
        Args:
            entity: Сущность для обновления
        Returns:
            T: Обновленная сущность
        Raises:
            EntityUpdateError: Ошибка обновления
            EntityNotFoundError: Сущность не найдена
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """
        Удаление сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: True если удалено успешно
        Raises:
            EntityNotFoundError: Сущность не найдена
        """
        pass

    @abstractmethod
    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """
        Мягкое удаление сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: True если удалено успешно
        """
        pass

    @abstractmethod
    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """
        Восстановление мягко удаленной сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: True если восстановлено успешно
        """
        pass

    @abstractmethod
    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[T]:
        """
        Поиск сущностей по фильтрам.
        Args:
            filters: Список фильтров
            options: Опции запроса
        Returns:
            List[T]: Список найденных сущностей
        """
        pass

    @abstractmethod
    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[T]:
        """
        Поиск одной сущности по фильтрам.
        Args:
            filters: Список фильтров
        Returns:
            Optional[T]: Найденная сущность или None
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """
        Проверка существования сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: True если сущность существует
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """
        Подсчет количества сущностей.
        Args:
            filters: Опциональные фильтры
        Returns:
            int: Количество сущностей
        """
        pass

    @abstractmethod
    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[T]:
        """
        Потоковое получение сущностей.
        Args:
            options: Опции запроса
            batch_size: Размер батча
        Yields:
            T: Сущности по одной
        """
        pass

    # ============================================================================
    # ТРАНЗАКЦИИ
    # ============================================================================
    @abstractmethod
    @asynccontextmanager
    def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """
        Контекстный менеджер для транзакций.
        Yields:
            TransactionProtocol: Объект транзакции
        """
        pass

    @abstractmethod
    async def execute_in_transaction(
        self, 
        operation: Callable[..., Any], 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """
        Выполнение операции в транзакции.
        Args:
            operation: Операция для выполнения
            *args: Аргументы операции
            **kwargs: Ключевые аргументы операции
        Returns:
            Any: Результат операции
        """
        pass

    # ============================================================================
    # ПАКЕТНЫЕ ОПЕРАЦИИ
    # ============================================================================
    @abstractmethod
    async def bulk_save(self, entities: List[T]) -> BulkOperationResult:
        """
        Пакетное сохранение сущностей.
        Args:
            entities: Список сущностей
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def bulk_update(self, entities: List[T]) -> BulkOperationResult:
        """
        Пакетное обновление сущностей.
        Args:
            entities: Список сущностей
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """
        Пакетное удаление сущностей.
        Args:
            entity_ids: Список ID сущностей
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def bulk_upsert(
        self, entities: List[T], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """
        Пакетное upsert сущностей.
        Args:
            entities: Список сущностей
            conflict_fields: Поля для разрешения конфликтов
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    # ============================================================================
    # КЭШИРОВАНИЕ
    # ============================================================================
    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[T]:
        """
        Получение из кэша.
        Args:
            key: Ключ кэша
        Returns:
            Optional[T]: Значение из кэша или None
        """
        cache_key = CacheKey(str(key))
        if cache_key not in self._cache:
            self._cache_misses += 1
            return None
        # Проверка TTL
        if cache_key in self._cache_ttl:
            if datetime.now() > self._cache_ttl[cache_key]:
                await self.invalidate_cache(key)
                self._cache_misses += 1
                return None
        self._cache_hits += 1
        return self._cache[cache_key]

    async def set_cache(
        self, key: Union[UUID, str], entity: T, ttl: Optional[int] = None
    ) -> None:
        """
        Установка в кэш.
        Args:
            key: Ключ кэша
            entity: Сущность для кэширования
            ttl: Время жизни в секундах
        """
        cache_key = CacheKey(str(key))
        ttl_seconds = ttl or self._cache_ttl_seconds
        # Эвакуация при необходимости
        if len(self._cache) >= self._cache_max_size:
            await self._evict_cache()
        self._cache[cache_key] = entity
        self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """
        Инвалидация кэша.
        Args:
            key: Ключ кэша
        """
        cache_key = CacheKey(str(key))
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_ttl:
            del self._cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        """Очистка всего кэша."""
        self._cache.clear()
        self._cache_ttl.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    async def _evict_cache(self) -> None:
        """Эвакуация кэша по стратегии."""
        if not self._cache:
            return
        if self._eviction_strategy == CacheEvictionStrategy.LRU:
            # LRU - удаляем самый старый по времени доступа
            oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
        elif self._eviction_strategy == CacheEvictionStrategy.FIFO:
            # FIFO - удаляем первый добавленный
            oldest_key = next(iter(self._cache.keys()))
        elif self._eviction_strategy == CacheEvictionStrategy.RANDOM:
            # Random - удаляем случайный
            import random

            oldest_key = random.choice(list(self._cache.keys()))
        else:
            # По умолчанию LRU
            oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
        await self.invalidate_cache(oldest_key)
        self._cache_evictions += 1

    # ============================================================================
    # МОНИТОРИНГ И МЕТРИКИ
    # ============================================================================
    async def get_repository_stats(self) -> RepositoryResponse:
        """
        Получение статистики репозитория.
        Returns:
            RepositoryResponse: Статистика репозитория
        """
        try:
            stats = {
                "state": self._state.value,
                "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
                "total_entities": self._metrics["total_entities"],
                "cache_hit_rate": self._get_cache_hit_rate(),
                "avg_query_time": self._metrics["avg_query_time"],
                "error_rate": self._get_error_rate(),
                "last_cleanup": self._last_cleanup.isoformat(),
                "memory_usage_mb": self._metrics["memory_usage_mb"],
                "disk_usage_mb": self._metrics["disk_usage_mb"],
            }
            return RepositoryResponse(
                success=True, data=stats, execution_time=0.0, cache_hit=False
            )
        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {e}")
            return RepositoryResponse(
                success=False,
                data={},
                error_message=str(e),
                execution_time=0.0,
                cache_hit=False,
            )

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Получение метрик производительности.
        Returns:
            PerformanceMetrics: Метрики производительности
        """
        return PerformanceMetrics(
            repository=self._metrics,
            cache=CacheMetrics(
                hits=self._cache_hits,
                misses=self._cache_misses,
                hit_rate=self._get_cache_hit_rate(),
                size=len(self._cache),
                max_size=self._cache_max_size,
                evictions=self._cache_evictions,
                avg_ttl=self._cache_ttl_seconds,
            ),
            transactions=self._transaction_metrics,
            custom_metrics={},
        )

    async def get_cache_stats(self) -> RepositoryResponse:
        """
        Получение статистики кэша.
        Returns:
            RepositoryResponse: Статистика кэша
        """
        try:
            cache_stats = {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": self._get_cache_hit_rate(),
                "size": len(self._cache),
                "max_size": self._cache_max_size,
                "evictions": self._cache_evictions,
                "avg_ttl": self._cache_ttl_seconds,
                "type": self._cache_type.value,
                "eviction_strategy": self._eviction_strategy.value,
            }
            return RepositoryResponse(
                success=True, data=cache_stats, execution_time=0.0, cache_hit=False
            )
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return RepositoryResponse(
                success=False,
                data={},
                error_message=str(e),
                execution_time=0.0,
                cache_hit=False,
            )

    async def health_check(self) -> HealthCheckResult:
        """
        Проверка здоровья репозитория.
        Returns:
            HealthCheckResult: Результат проверки здоровья
        """
        start_time = time.time()
        try:
            # Базовая проверка состояния
            if self._state != RepositoryState.CONNECTED:
                return HealthCheckResult(
                    status="unhealthy",
                    timestamp=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                    error_count=self._error_count,
                    last_error=self._last_error,
                    uptime_seconds=(
                        datetime.now() - self._startup_time
                    ).total_seconds(),
                    memory_usage_mb=self._metrics["memory_usage_mb"],
                    disk_usage_mb=self._metrics["disk_usage_mb"],
                    connection_status="disconnected",
                    cache_status="unknown",
                )
            # Проверка кэша
            cache_status = (
                "healthy" if len(self._cache) < self._cache_max_size else "degraded"
            )
            return HealthCheckResult(
                status="healthy",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_count=self._error_count,
                last_error=self._last_error,
                uptime_seconds=(datetime.now() - self._startup_time).total_seconds(),
                memory_usage_mb=self._metrics["memory_usage_mb"],
                disk_usage_mb=self._metrics["disk_usage_mb"],
                connection_status="connected",
                cache_status=cache_status,
            )
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status="unhealthy",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_count=self._error_count,
                last_error=str(e),
                uptime_seconds=(datetime.now() - self._startup_time).total_seconds(),
                memory_usage_mb=0.0,
                disk_usage_mb=0.0,
                connection_status="error",
                cache_status="error",
            )

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================================
    def _get_cache_hit_rate(self) -> float:
        """Получение hit rate кэша."""
        total_requests = self._cache_hits + self._cache_misses
        return self._cache_hits / total_requests if total_requests > 0 else 0.0

    def _get_error_rate(self) -> float:
        """Получение rate ошибок."""
        return self._error_count / max(self._operation_count, 1)

    def _update_metrics(self, operation_time: float) -> None:
        """Обновление метрик."""
        self._operation_count += 1
        self._query_times.append(operation_time)
        if self._query_times:
            self._metrics["avg_query_time"] = sum(self._query_times) / len(
                self._query_times
            )

    def _record_error(self, error: Exception) -> None:
        """Запись ошибки."""
        self._error_count += 1
        self._last_error = str(error)
        self.logger.error(f"Repository error: {error}")

    async def _background_cleanup(self) -> None:
        """Фоновая очистка."""
        cleanup_interval = self.config.get("cleanup_interval", 3600)  # 1 час
        while self._state != RepositoryState.SHUTTING_DOWN:
            try:
                await asyncio.sleep(cleanup_interval)
                # Очистка устаревших записей кэша
                now = datetime.now()
                expired_keys = [
                    key for key, ttl in self._cache_ttl.items() if now > ttl
                ]
                for key in expired_keys:
                    await self.invalidate_cache(key)
                # Обновление метрик
                self._last_cleanup = now
                self._metrics["last_cleanup"] = now.isoformat()
                self._metrics["uptime_seconds"] = (
                    now - self._startup_time
                ).total_seconds()
                self.logger.debug(
                    f"Background cleanup completed, removed {len(expired_keys)} expired cache entries"
                )
            except Exception as e:
                self.logger.error(f"Background cleanup failed: {e}")
                await asyncio.sleep(60)  # Пауза перед следующей попыткой

    async def _metrics_collector(self) -> None:
        """Сборщик метрик."""
        metrics_interval = 60  # 1 минута
        while self._state != RepositoryState.SHUTTING_DOWN:
            try:
                await asyncio.sleep(metrics_interval)
                # Обновление метрик производительности
                self._metrics["cache_hit_rate"] = self._get_cache_hit_rate()
                self._metrics["error_rate"] = self._get_error_rate()
                # Обновление метрик памяти (упрощенная реализация)
                import psutil

                process = psutil.Process()
                memory_info = process.memory_info()
                self._metrics["memory_usage_mb"] = memory_info.rss / 1024 / 1024
                self.logger.debug("Metrics collection completed")
            except Exception as e:
                self.logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(60)

    async def close(self) -> None:
        """Закрытие репозитория."""
        self._state = RepositoryState.SHUTTING_DOWN
        await self.clear_cache()
        self.logger.info(f"{self.__class__.__name__} closed successfully")


# Исключения
class RepositoryError(Exception):
    """Базовое исключение для репозиториев."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause
        self.message = message


class EntityNotFoundError(RepositoryError):
    """Исключение при отсутствии сущности."""

    pass


class ValidationError(RepositoryError):
    """Исключение при ошибке валидации."""

    pass


class RepositoryConnectionError(RepositoryError):
    """Исключение при ошибке соединения с репозиторием."""

    pass
