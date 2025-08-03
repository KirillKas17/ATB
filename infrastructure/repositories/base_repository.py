"""
Базовый репозиторий с унифицированными паттернами для Infrastructure Layer.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Generic, List, Optional, TypeVar, Union, Coroutine
from uuid import UUID
from contextlib import asynccontextmanager

from domain.protocols.repository_protocol import (
    BulkOperationResult,
    QueryFilter,
    QueryOptions,
    RepositoryProtocol,
    RepositoryResponse,
    RepositoryState,
    TransactionProtocol,
)
from domain.types.repository_types import RepositoryOperation, SortOrder, Pagination
from infrastructure.shared.cache import (
    CacheConfig,
    CacheEvictionStrategy,
    CacheProtocol,
    CacheType,
)
from infrastructure.shared.exceptions import (
    DataIntegrityError,
    RepositoryError,
    ValidationError,
)
from infrastructure.shared.logging import RepositoryLogger
from domain.types.protocol_types import HealthCheckDict

T = TypeVar("T")


@dataclass
class RepositoryMetrics:
    """Метрики репозитория."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time_ms: float = 0.0
    last_operation_time: Optional[datetime] = None
    startup_time: datetime = field(default_factory=datetime.now)

    def record_operation(self, success: bool, response_time_ms: float) -> None:
        """Запись операции."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        # Обновляем среднее время ответа
        if self.total_operations == 1:
            self.avg_response_time_ms = response_time_ms
        else:
            self.avg_response_time_ms = (
                self.avg_response_time_ms * (self.total_operations - 1)
                + response_time_ms
            ) / self.total_operations
        self.last_operation_time = datetime.now()

    def record_cache_hit(self) -> None:
        """Запись кэш-хита."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Запись кэш-промаха."""
        self.cache_misses += 1

    def get_success_rate(self) -> float:
        """Получение процента успешных операций."""
        return (
            self.successful_operations / self.total_operations
            if self.total_operations > 0
            else 0.0
        )

    def get_cache_hit_rate(self) -> float:
        """Получение процента кэш-хитов."""
        total_cache_ops = self.cache_hits + self.cache_misses
        return self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0


class BaseRepository(RepositoryProtocol, Generic[T], ABC):
    """
    Базовый репозиторий с унифицированными паттернами.
    Предоставляет общую функциональность для всех репозиториев:
    - Кэширование
    - Метрики
    - Логирование
    - Валидация
    - Обработка ошибок
    - Транзакции
    """

    def __init__(
        self,
        name: str,
        cache_config: Optional[CacheConfig] = None,
        enable_metrics: bool = True,
        enable_logging: bool = True,
    ):
        self.name = name
        self.cache_config = cache_config or CacheConfig()
        self.enable_metrics = enable_metrics
        self.enable_logging = enable_logging
        # Инициализация компонентов
        self._logger = RepositoryLogger(name) if enable_logging else None
        self._metrics = RepositoryMetrics() if enable_metrics else None
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._state = RepositoryState.DISCONNECTED
        self._lock = asyncio.Lock()
        # Инициализация кэша
        if self.cache_config.cache_type != CacheType.MEMORY:
            if self._logger:
                self._logger.warning(
                    f"Cache type {self.cache_config.cache_type} not implemented, using memory cache"
                )
        self._initialize_cache()
        self._state = RepositoryState.CONNECTED
        if self._logger:
            self._logger.info(f"Initialized repository: {name}")

    def _initialize_cache(self) -> None:
        """Инициализация кэша."""
        try:
            from infrastructure.shared.cache import MemoryCache
            cache_instance = MemoryCache(self.cache_config)
            # Приводим к совместимому типу с базовым классом
            # self._cache уже определен в __init__, поэтому не переопределяем
            if self._logger:
                self._logger.info(f"Initialized cache for repository: {self.name}")
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to initialize cache: {e}")
            # self._cache = {}  # Убираем дублирование

    @abstractmethod
    async def _save_entity(self, entity: T) -> T:
        """Сохранение сущности (абстрактный метод)."""
        pass

    @abstractmethod
    async def _get_entity_by_id(self, entity_id: Union[UUID, str]) -> Optional[T]:
        """Получение сущности по ID (абстрактный метод)."""
        pass

    @abstractmethod
    async def _get_all_entities(self) -> List[T]:
        """Получение всех сущностей (абстрактный метод)."""
        pass

    @abstractmethod
    async def _delete_entity(self, entity_id: Union[UUID, str]) -> bool:
        """Удаление сущности (абстрактный метод)."""
        pass

    @abstractmethod
    async def _find_entities_by_filters(self, filters: List[QueryFilter]) -> List[T]:
        """Поиск сущностей по фильтрам (абстрактный метод)."""
        pass

    async def save(self, entity: T) -> T:
        """Сохранение сущности с кэшированием и метриками."""
        start_time = datetime.now()
        success = False
        try:
            # Валидация сущности
            await self._validate_entity(entity)
            # Сохранение
            result = await self._save_entity(entity)
            # Инвалидация кэша
            if self._cache:
                cache_key = self._generate_cache_key(
                    "entity", str(getattr(entity, "id", hash(entity)))
                )
                if hasattr(self._cache, 'delete'):
                    await self._cache.delete(cache_key)
            success = True
            if self._logger:
                self._logger.log_database_operation(
                    "save", self._get_entity_type_name(), 1
                )
            return result
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saving entity: {e}")
            raise RepositoryError(f"Failed to save entity: {str(e)}")
        finally:
            if self._metrics:
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.record_operation(success, response_time_ms)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[T]:
        """Получение сущности по ID с кэшированием."""
        start_time = datetime.now()
        success = False
        try:
            # Проверка кэша
            cache_key = self._generate_cache_key("entity", str(entity_id))
            if self._cache and hasattr(self._cache, 'get'):
                cached_entity: Optional[T] = await self._cache.get(cache_key)
                if cached_entity:
                    if self._metrics:
                        self._metrics.record_cache_hit()
                    return cached_entity

            # Получение из хранилища
            entity = await self._get_entity_by_id(entity_id)
            if entity and self._cache and hasattr(self._cache, 'set'):
                ttl = getattr(self.cache_config, 'ttl_seconds', 300)
                await self._cache.set(cache_key, entity, ttl)
            
            success = True
            if self._metrics:
                self._metrics.record_cache_miss()
            return entity
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error getting entity by ID: {e}")
            raise RepositoryError(f"Failed to get entity: {str(e)}")
        finally:
            if self._metrics:
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.record_operation(success, response_time_ms)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[T]:
        """Получение всех сущностей с применением опций."""
        start_time = datetime.now()
        success = False
        try:
            entities = await self._get_all_entities()
            
            if options:
                entities = await self._apply_query_options(entities, options)
            
            success = True
            return entities
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error getting all entities: {e}")
            raise RepositoryError(f"Failed to get all entities: {str(e)}")
        finally:
            if self._metrics:
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.record_operation(success, response_time_ms)

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удаление сущности."""
        start_time = datetime.now()
        success = False
        try:
            result = await self._delete_entity(entity_id)
            if result and self._cache and hasattr(self._cache, 'delete'):
                cache_key = self._generate_cache_key("entity", str(entity_id))
                await self._cache.delete(cache_key)
            
            success = True
            return result
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error deleting entity: {e}")
            raise RepositoryError(f"Failed to delete entity: {str(e)}")
        finally:
            if self._metrics:
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.record_operation(success, response_time_ms)

    async def update(self, entity: T) -> T:
        """Обновление сущности."""
        start_time = datetime.now()
        success = False
        try:
            await self._validate_entity(entity)
            result = await self._save_entity(entity)
            
            # Инвалидация кэша
            if self._cache and hasattr(self._cache, 'delete'):
                cache_key = self._generate_cache_key("entity", str(getattr(entity, "id", hash(entity))))
                await self._cache.delete(cache_key)
            
            success = True
            return result
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error updating entity: {e}")
            raise RepositoryError(f"Failed to update entity: {str(e)}")
        finally:
            if self._metrics:
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.record_operation(success, response_time_ms)

    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[T]:
        """Поиск сущностей по фильтрам."""
        start_time = datetime.now()
        success = False
        try:
            entities = await self._find_entities_by_filters(filters)
            
            if options:
                entities = await self._apply_query_options(entities, options)
            
            success = True
            return entities
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error finding entities: {e}")
            raise RepositoryError(f"Failed to find entities: {str(e)}")
        finally:
            if self._metrics:
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.record_operation(success, response_time_ms)

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчет количества сущностей."""
        try:
            if filters:
                entities = await self._find_entities_by_filters(filters)
                return len(entities)
            else:
                entities = await self._get_all_entities()
                return len(entities)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error counting entities: {e}")
            raise RepositoryError(f"Failed to count entities: {str(e)}")

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверка существования сущности."""
        try:
            entity = await self._get_entity_by_id(entity_id)
            return entity is not None
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error checking entity existence: {e}")
            raise RepositoryError(f"Failed to check entity existence: {str(e)}")

    async def bulk_save(self, entities: List[T]) -> BulkOperationResult:
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity in entities:
            try:
                await self.save(entity)
                entity_id = getattr(entity, 'id', str(hash(entity)))
                processed_ids.append(entity_id)
            except Exception as e:
                errors.append({"entity_id": str(getattr(entity, 'id', hash(entity))), "error": str(e)})
        
        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=len(processed_ids),
            error_count=len(errors)
        )

    async def bulk_update(self, entities: List[T]) -> BulkOperationResult:
        """Пакетное обновление сущностей."""
        start_time = datetime.now()
        success_count = 0
        error_count = 0
        errors = []
        processed_ids = []

        for entity in entities:
            try:
                await self.update(entity)
                success_count += 1
                processed_ids.append(getattr(entity, "id", hash(entity)))
            except Exception as e:
                error_count += 1
                errors.append({"entity_id": str(getattr(entity, "id", hash(entity))), "error": str(e)})

        execution_time = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
            processed_ids=[UUID(uid) if isinstance(uid, str) and len(uid) == 36 else uid for uid in processed_ids if isinstance(uid, (str, UUID))],
            execution_time=execution_time
        )

    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """Пакетное удаление сущностей."""
        start_time = datetime.now()
        success_count = 0
        error_count = 0
        errors = []
        processed_ids = []

        for entity_id in entity_ids:
            try:
                if await self.delete(entity_id):
                    success_count += 1
                    processed_ids.append(entity_id)
                else:
                    error_count += 1
                    errors.append({"entity_id": str(entity_id), "error": "Entity not found"})
            except Exception as e:
                error_count += 1
                errors.append({"entity_id": str(entity_id), "error": str(e)})

        execution_time = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
            processed_ids=processed_ids,
            execution_time=execution_time
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Транзакция."""
        class MockTransaction:
            async def __aenter__(self) -> "MockTransaction":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def commit(self) -> None:
                pass

            async def rollback(self) -> None:
                pass

            async def is_active(self) -> bool:
                return True

        transaction = MockTransaction()
        yield transaction

    async def execute_in_transaction(
        self, operation: RepositoryOperation, *args: Any, **kwargs: Any
    ) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as txn:
            return await operation(*args, **kwargs)

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получение статистики репозитория."""
        try:
            stats = {
                "total_entities": len(await self.get_all()),
                "cache_hit_rate": self._metrics.get_cache_hit_rate() if self._metrics else 0.0,
                "avg_query_time": self._metrics.avg_response_time_ms if self._metrics else 0.0,
                "error_rate": 1.0 - (self._metrics.get_success_rate() if self._metrics else 1.0),
                "last_cleanup": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self._metrics.startup_time).total_seconds() if self._metrics else 0.0,
                "memory_usage_mb": 0.0,  # Placeholder
                "disk_usage_mb": 0.0,  # Placeholder
            }
            return RepositoryResponse(success=True, data=stats)
        except Exception as e:
            return RepositoryResponse(success=False, data={"error": str(e)})

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            # Простая проверка - попытка получить количество сущностей
            await self.count()
            return HealthCheckDict(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                response_time=0.001,
                error_count=0,
                last_error=None,
                uptime=0.0
            )
        except Exception as e:
            return HealthCheckDict(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                response_time=0.0,
                error_count=1,
                last_error=str(e),
                uptime=0.0
            )

    def _generate_cache_key(self, *args: Any) -> str:
        """Генерация ключа кэша."""
        return f"{self.name}:{':'.join(str(arg) for arg in args)}"

    def _get_entity_type_name(self) -> str:
        """Получение имени типа сущности."""
        return self.__class__.__name__.replace("Repository", "").lower()

    async def _validate_entity(self, entity: T) -> None:
        """Валидация сущности."""
        if entity is None:
            raise ValidationError("Entity cannot be None")

    async def _apply_query_options(
        self, entities: List[T], options: QueryOptions
    ) -> List[T]:
        """Применение опций запроса к сущностям."""
        # Сортировка
        if options.sort_orders:
            entities = self._sort_entities(entities, options.sort_orders)
        
        # Пагинация
        if options.pagination:
            entities = self._paginate_entities(entities, options.pagination)
        
        return entities

    def _sort_entities(
        self, entities: List[T], sort_orders: List[SortOrder]
    ) -> List[T]:
        """Сортировка сущностей."""
        for sort_order in reversed(sort_orders):
            reverse = sort_order.direction == "desc"
            entities.sort(
                key=lambda x: getattr(x, sort_order.field, 0),
                reverse=reverse
            )
        return entities

    def _paginate_entities(
        self, entities: List[T], pagination: Pagination
    ) -> List[T]:
        """Пагинация сущностей."""
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        return entities[start:end]
