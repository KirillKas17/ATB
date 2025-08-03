"""
Базовая реализация репозитория для устранения дублирования кода.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID
import logging

from loguru import logger

from domain.exceptions.base_exceptions import (
    EntityDeleteError,
    EntityNotFoundError,
    EntitySaveError,
    RepositoryError,
)
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    QueryFilter,
    QueryOptions,
    RepositoryProtocol,
    RepositoryResponse,
    RepositoryState,
)
from domain.types.common_types import EntityId, JsonData, ValidationResult

T = TypeVar("T")


@dataclass
class RepositoryMetrics:
    """Метрики репозитория."""

    operation_count: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    last_operation: Optional[datetime] = None


class BaseRepositoryImpl(RepositoryProtocol, ABC, Generic[T]):
    """
    Базовая реализация репозитория с устранением дублирования.
    Предоставляет общую логику для всех репозиториев:
    - Кэширование с TTL
    - Валидация сущностей
    - Обработка ошибок
    - Метрики и мониторинг
    - Транзакции
    - Пакетные операции
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация базового репозитория."""
        self.config = config or {}
        # Используем loguru logger напрямую, так как он совместим с logging.Logger
        self.logger: Any = logger.bind(module=self.__class__.__name__)
        # Кэширование
        self._cache: Dict[Union[UUID, str], T] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._cache_max_size = self.config.get("cache_max_size", 1000)
        self._cache_ttl_seconds = self.config.get("cache_ttl", 300)
        # Состояние
        self._state = RepositoryState.DISCONNECTED
        self._startup_time = datetime.now()
        self._last_cleanup = datetime.now()
        # Метрики
        self._metrics = RepositoryMetrics()
        # Инициализация
        self._initialize_repository()

    def _initialize_repository(self) -> None:
        """Инициализация репозитория."""
        try:
            self._perform_cleanup()
            self._state = RepositoryState.CONNECTED
            self.logger.info(
                f"Repository {self.__class__.__name__} initialized successfully"
            )
        except Exception as e:
            self._state = RepositoryState.ERROR
            self.logger.error(f"Failed to initialize repository: {e}")
            raise RepositoryError(f"Repository initialization failed: {e}")

    @property
    def state(self) -> RepositoryState:
        """Текущее состояние репозитория."""
        return self._state

    @property
    def is_healthy(self) -> bool:
        """Проверка здоровья репозитория."""
        return self._state == RepositoryState.CONNECTED

    # ============================================================================
    # КЭШИРОВАНИЕ
    # ============================================================================
    def _generate_cache_key(self, prefix: str, identifier: str) -> str:
        """Генерация ключа кэша."""
        return f"{self.__class__.__name__}:{prefix}:{identifier}"

    def _get_from_cache(self, key: Union[UUID, str]) -> Optional[T]:
        """Получение из кэша."""
        if key not in self._cache:
            self._metrics.cache_misses += 1
            return None
        # Проверка TTL
        if key in self._cache_ttl:
            if datetime.now() > self._cache_ttl[key]:
                del self._cache[key]
                del self._cache_ttl[key]
                self._metrics.cache_misses += 1
                return None
        self._metrics.cache_hits += 1
        return self._cache[key]

    def _set_cache(self, key: Union[UUID, str], value: T, ttl: Optional[int] = None) -> None:
        """Установка в кэш."""
        # Очистка при превышении размера
        if len(self._cache) >= self._cache_max_size:
            self._evict_cache()
        self._cache[key] = value
        ttl_seconds = ttl or self._cache_ttl_seconds
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl_seconds)

    def _evict_cache(self) -> None:
        """Вытеснение из кэша (LRU)."""
        if not self._cache:
            return
        # Удаляем самый старый элемент
        oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
        del self._cache[oldest_key]
        del self._cache_ttl[oldest_key]

    def _invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидация кэша."""
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_ttl:
            del self._cache_ttl[key]

    def _clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()
        self._cache_ttl.clear()

    # ============================================================================
    # ВАЛИДАЦИЯ
    # ============================================================================
    async def _validate_entity(self, entity: T) -> ValidationResult:
        """Валидация сущности."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            # Базовая валидация
            if entity is None:
                errors.append("Entity cannot be None")
                return ValidationResult(
                    is_valid=False, errors=errors, warnings=warnings
                )
            # Проверка обязательных полей
            required_fields = self._get_required_fields()
            for field in required_fields:
                if not hasattr(entity, field) or getattr(entity, field) is None:
                    errors.append(f"Required field '{field}' is missing or None")
            # Специфичная валидация
            entity_errors, entity_warnings = await self._validate_entity_specific(
                entity
            )
            errors.extend(entity_errors)
            warnings.extend(entity_warnings)
            return ValidationResult(
                is_valid=len(errors) == 0, errors=errors, warnings=warnings
            )
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    def _get_required_fields(self) -> List[str]:
        """Получение обязательных полей (переопределяется в наследниках)."""
        return ["id"]

    async def _validate_entity_specific(self, entity: T) -> tuple[List[str], List[str]]:
        """Специфичная валидация сущности (переопределяется в наследниках)."""
        return [], []

    # ============================================================================
    # ОБРАБОТКА ОШИБОК
    # ============================================================================
    def _handle_operation_error(self, operation: str, error: Exception) -> None:
        """Обработка ошибки операции."""
        self._metrics.error_count += 1
        self._metrics.last_operation = datetime.now()
        error_msg = f"Operation '{operation}' failed: {str(error)}"
        self.logger.error(error_msg)
        # Определение типа ошибки
        if isinstance(error, (EntityNotFoundError, EntitySaveError, EntityDeleteError)):
            raise error
        elif isinstance(error, ValueError):
            raise EntitySaveError(f"Validation error: {str(error)}")
        else:
            raise RepositoryError(f"Repository operation failed: {str(error)}")

    def _record_operation(self, operation: str, success: bool, duration: float) -> None:
        """Запись метрик операции."""
        self._metrics.operation_count += 1
        if not success:
            self._metrics.error_count += 1
        self._metrics.last_operation = datetime.now()
        # Обновление среднего времени ответа
        if self._metrics.operation_count > 0:
            self._metrics.avg_response_time = (
                (self._metrics.avg_response_time * (self._metrics.operation_count - 1) + duration)
                / self._metrics.operation_count
            )

    # ============================================================================
    # ОСНОВНЫЕ ОПЕРАЦИИ
    # ============================================================================
    async def save(self, entity: T) -> T:
        """Сохранение сущности."""
        start_time = datetime.now()
        try:
            # Валидация
            validation_result = await self._validate_entity(entity)
            if not validation_result["is_valid"]:
                raise EntitySaveError(f"Validation failed: {validation_result['errors']}")
            
            # Сохранение
            saved_entity = await self._save_entity_impl(entity)
            
            # Кэширование
            entity_id = self._get_entity_id(saved_entity)
            cache_key = self._generate_cache_key("entity", str(entity_id))
            self._set_cache(cache_key, saved_entity)
            
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("save", True, duration)
            return saved_entity
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("save", False, duration)
            self._handle_operation_error("save", e)
            raise

    async def get_by_id(self, entity_id: EntityId) -> Optional[T]:
        """Получение сущности по ID."""
        start_time = datetime.now()
        try:
            # Проверка кэша
            cache_key = self._generate_cache_key("entity", str(entity_id))
            cached_entity = self._get_from_cache(cache_key)
            if cached_entity is not None:
                duration = (datetime.now() - start_time).total_seconds()
                self._record_operation("get_by_id", True, duration)
                return cached_entity
            
            # Получение из хранилища
            entity = await self._get_entity_by_id_impl(entity_id)
            if entity is not None:
                self._set_cache(cache_key, entity)
            
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("get_by_id", True, duration)
            return entity
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("get_by_id", False, duration)
            self._handle_operation_error("get_by_id", e)
            raise

    async def update(self, entity: T) -> T:
        """Обновление сущности."""
        start_time = datetime.now()
        try:
            # Валидация
            validation_result = await self._validate_entity(entity)
            if not validation_result["is_valid"]:
                raise EntitySaveError(f"Validation failed: {validation_result['errors']}")
            
            # Обновление
            updated_entity = await self._update_entity_impl(entity)
            
            # Инвалидация кэша
            entity_id = self._get_entity_id(updated_entity)
            cache_key = self._generate_cache_key("entity", str(entity_id))
            self._invalidate_cache(cache_key)
            
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("update", True, duration)
            return updated_entity
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("update", False, duration)
            self._handle_operation_error("update", e)
            raise

    async def delete(self, entity_id: EntityId) -> bool:
        """Удаление сущности."""
        start_time = datetime.now()
        try:
            # Удаление
            success = await self._delete_entity_impl(entity_id)
            
            # Инвалидация кэша
            if success:
                cache_key = self._generate_cache_key("entity", str(entity_id))
                self._invalidate_cache(cache_key)
            
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("delete", success, duration)
            return success
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._record_operation("delete", False, duration)
            self._handle_operation_error("delete", e)
            raise

    async def exists(self, entity_id: EntityId) -> bool:
        """Проверка существования сущности."""
        try:
            entity = await self.get_by_id(entity_id)
            return entity is not None
        except Exception as e:
            self.logger.error(f"Error checking entity existence: {e}")
            return False

    # ============================================================================
    # ПАКЕТНЫЕ ОПЕРАЦИИ
    # ============================================================================
    async def bulk_save(self, entities: List[T]) -> BulkOperationResult:
        """Пакетное сохранение сущностей."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity in entities:
            try:
                saved_entity = await self.save(entity)
                entity_id = self._get_entity_id(saved_entity)
                processed_ids.append(entity_id)
            except Exception as e:
                errors.append({"entity_id": str(self._get_entity_id(entity)), "error": str(e)})
        
        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=len(processed_ids),
            error_count=len(errors)
        )

    async def bulk_update(self, entities: List[T]) -> BulkOperationResult:
        """Пакетное обновление сущностей."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity in entities:
            try:
                updated_entity = await self.update(entity)
                entity_id = self._get_entity_id(updated_entity)
                processed_ids.append(entity_id)
            except Exception as e:
                errors.append({"entity_id": str(self._get_entity_id(entity)), "error": str(e)})
        
        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=len(processed_ids),
            error_count=len(errors)
        )

    async def bulk_delete(self, entity_ids: List[EntityId]) -> BulkOperationResult:
        """Пакетное удаление сущностей."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for entity_id in entity_ids:
            try:
                success = await self.delete(entity_id)
                if success:
                    processed_ids.append(entity_id)
                else:
                    errors.append({"entity_id": str(entity_id), "error": "Entity not found"})
            except Exception as e:
                errors.append({"entity_id": str(entity_id), "error": str(e)})
        
        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=len(processed_ids),
            error_count=len(errors)
        )

    # ============================================================================
    # АБСТРАКТНЫЕ МЕТОДЫ
    # ============================================================================
    @abstractmethod
    async def _save_entity_impl(self, entity: T) -> T:
        """Реализация сохранения сущности."""
        pass

    @abstractmethod
    async def _get_entity_by_id_impl(self, entity_id: EntityId) -> Optional[T]:
        """Реализация получения сущности по ID."""
        pass

    @abstractmethod
    async def _update_entity_impl(self, entity: T) -> T:
        """Реализация обновления сущности."""
        pass

    @abstractmethod
    async def _delete_entity_impl(self, entity_id: EntityId) -> bool:
        """Реализация удаления сущности."""
        pass

    @abstractmethod
    def _get_entity_id(self, entity: T) -> EntityId:
        """Получение ID сущности."""
        pass

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================================
    def _perform_cleanup(self) -> None:
        """Выполнение очистки."""
        self._cleanup_expired_cache()

    def _cleanup_expired_cache(self) -> None:
        """Очистка истекшего кэша."""
        current_time = datetime.now()
        expired_keys = [
            key for key, expiry in self._cache_ttl.items()
            if current_time > expiry
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]

    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик репозитория."""
        return {
            "operation_count": self._metrics.operation_count,
            "error_count": self._metrics.error_count,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "avg_response_time": self._metrics.avg_response_time,
            "last_operation": self._metrics.last_operation.isoformat() if self._metrics.last_operation else None,
            "state": self._state.value,
            "is_healthy": self.is_healthy,
        }
