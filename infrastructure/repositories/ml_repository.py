"""
Реализация репозитория ML моделей.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Coroutine, Callable, cast, AsyncContextManager
from uuid import UUID, uuid4
from decimal import Decimal
from contextlib import asynccontextmanager
import json
from contextlib import AbstractAsyncContextManager, _AsyncGeneratorContextManager

from domain.entities.ml import Model, Prediction, ModelType, ModelStatus
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    MLRepositoryProtocol,
    QueryFilter,
    QueryOptions,
    RepositoryResponse,
    RepositoryState,
    PerformanceMetricsDict,
    HealthCheckDict,
    TransactionProtocol,
    QueryOperator,
)
from domain.value_objects.trading_pair import TradingPair
from domain.exceptions.protocol_exceptions import EntityUpdateError
from domain.exceptions.base_exceptions import RepositoryError
from domain.value_objects.currency import Currency


class InMemoryMLRepository(MLRepositoryProtocol):
    """
    In-memory реализация репозитория ML моделей.

    Особенности:
    - Хранение в памяти с кэшированием
    - Индексация по типам и статусам
    - Метрики производительности
    - Мониторинг и health-check
    """

    def __init__(self) -> None:
        self.models: Dict[UUID, Model] = {}
        self.predictions: Dict[UUID, Prediction] = {}
        self.models_by_type: Dict[str, List[UUID]] = defaultdict(list)
        self.models_by_status: Dict[str, List[UUID]] = defaultdict(list)
        self.models_by_trading_pair: Dict[str, List[UUID]] = defaultdict(list)
        self.predictions_by_model: Dict[UUID, List[UUID]] = defaultdict(list)
        self.model_metrics: Dict[UUID, List[Dict[str, Any]]] = defaultdict(list)
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.cache_max_size = 1000
        self.cache_ttl_seconds = 300
        self.metrics = {
            "total_models": 0,
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup": datetime.now(),
        }
        self._state = RepositoryState.CONNECTED
        self.startup_time = datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("InMemoryMLRepository initialized")

    async def save_model(self, model: Model) -> Model:
        """Сохранить модель."""
        try:
            self.models[model.id] = model
            self.models_by_type[str(model.model_type)].append(model.id)
            self.models_by_status[model.status.value].append(model.id)
            self.models_by_trading_pair[str(model.trading_pair)].append(model.id)
            current_total = self.metrics.get("total_models", 0)
            if isinstance(current_total, (int, float, str)):
                self.metrics["total_models"] = int(current_total) + 1
            else:
                self.metrics["total_models"] = 1
            await self.invalidate_cache(f"model:{model.id}")
            return model
        except Exception:
            raise EntityUpdateError(entity_type="Model", entity_id=str(model.id), message=f"Failed to save model: {model.id}")

    async def get_model(self, model_id: Union[UUID, str]) -> Optional[Model]:
        """Получить модель."""
        if isinstance(model_id, str):
            try:
                model_id = UUID(model_id)
            except ValueError:
                return None
        cache_key = f"model:{model_id}"
        cached = await self.get_from_cache(cache_key)
        if cached:
            cache_hits_raw = self.metrics.get("cache_hits", 0)
            if isinstance(cache_hits_raw, (int, float, str)):
                self.metrics["cache_hits"] = int(cache_hits_raw) + 1
            else:
                self.metrics["cache_hits"] = 1
            return cached
        else:
            cache_misses_raw = self.metrics.get("cache_misses", 0)
            if isinstance(cache_misses_raw, (int, float, str)):
                self.metrics["cache_misses"] = int(cache_misses_raw) + 1
            else:
                self.metrics["cache_misses"] = 1
            model = self.models.get(model_id)
            if model:
                await self.set_cache(cache_key, model)
            return model

    async def get_models_by_type(self, model_type: str) -> List[Model]:
        """Получить модели по типу."""
        model_ids = self.models_by_type.get(model_type, [])
        return [self.models[model_id] for model_id in model_ids if model_id in self.models]

    async def get_best_model(self, model_type: str) -> Optional[Model]:
        """Получить лучшую модель по типу."""
        models = await self.get_models_by_type(model_type)
        if not models:
            return None
        return max(models, key=lambda m: m.accuracy if hasattr(m, 'accuracy') else 0.0)

    async def save(self, entity: Model) -> Model:
        """Сохранить сущность."""
        return await self.save_model(entity)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Model]:
        """Получить сущность по ID."""
        return await self.get_model(entity_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Model]:
        """Получить все сущности."""
        models = list(self.models.values())
        if options and options.pagination:
            start = (options.pagination.page - 1) * options.pagination.page_size
            end = start + options.pagination.page_size
            models = models[start:end]
        return models

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить сущность."""
        if isinstance(entity_id, str):
            try:
                entity_id = UUID(entity_id)
            except ValueError:
                return False
        
        if entity_id in self.models:
            model = self.models[entity_id]
            # Удаляем из индексов
            self.models_by_type[str(model.model_type)].remove(entity_id)
            self.models_by_status[model.status.value].remove(entity_id)
            self.models_by_trading_pair[str(model.trading_pair)].remove(entity_id)
            # Удаляем из основного хранилища
            del self.models[entity_id]
            # Инвалидируем кэш
            await self.invalidate_cache(f"model:{entity_id}")
            return True
        return False

    async def get_active_models(self) -> List[Model]:
        """Получить активные модели."""
        active_ids = self.models_by_status.get(ModelStatus.ACTIVE.value, [])
        return [self.models[model_id] for model_id in active_ids if model_id in self.models]

    async def get_model_by_trading_pair(self, trading_pair: str) -> List[Model]:
        """Получить модели по торговой паре."""
        model_ids = self.models_by_trading_pair.get(trading_pair, [])
        return [self.models[model_id] for model_id in model_ids if model_id in self.models]

    async def save_prediction(self, prediction: Prediction) -> Prediction:
        """Сохранить предсказание."""
        self.predictions[prediction.id] = prediction
        self.predictions_by_model[prediction.model_id].append(prediction.id)
        return prediction

    async def get_predictions_by_model(self, model_id: UUID, limit: int = 100) -> List[Prediction]:
        """Получить предсказания по модели."""
        prediction_ids = self.predictions_by_model.get(model_id, [])[-limit:]
        return [self.predictions[pred_id] for pred_id in prediction_ids if pred_id in self.predictions]

    async def get_latest_prediction(self, model_id: UUID) -> Optional[Prediction]:
        """Получить последнее предсказание модели."""
        prediction_ids = self.predictions_by_model.get(model_id, [])
        if not prediction_ids:
            return None
        latest_id = prediction_ids[-1]
        return self.predictions.get(latest_id)

    async def update_model_metrics(self, model_id: UUID, metrics: Dict[str, float]) -> bool:
        """Обновить метрики модели."""
        if model_id in self.models:
            self.model_metrics[model_id].append(metrics)
            return True
        return False

    async def update(self, entity: Model) -> Model:
        """Обновить сущность."""
        return await self.save_model(entity)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление сущности."""
        return await self.delete(entity_id)

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить сущность."""
        # В in-memory реализации просто возвращаем True
        return True

    async def find_by(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> List[Model]:
        """Найти сущности по фильтрам."""
        models = list(self.models.values())
        for filter_item in filters:
            models = [m for m in models if self._matches_filter(m, filter_item)]
        return models

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Model]:
        """Найти одну сущность по фильтрам."""
        models = await self.find_by(filters)
        return models[0] if models else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование сущности."""
        if isinstance(entity_id, str):
            try:
                entity_id = UUID(entity_id)
            except ValueError:
                return False
        return entity_id in self.models

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество сущностей."""
        if filters:
            models = await self.find_by(filters)
            return len(models)
        return len(self.models)

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[Model]:
        """Потоковое чтение сущностей."""
        models = list(self.models.values())
        
        for i in range(0, len(models), batch_size):
            batch = models[i:i + batch_size]
            for model in batch:
                yield model
            await asyncio.sleep(0)  # Даем возможность другим задачам выполниться
        
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        class MockTransaction(TransactionProtocol):
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

        yield MockTransaction()

    async def execute_in_transaction(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as txn:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[Model]) -> BulkOperationResult:
        """Пакетное сохранение сущностей."""
        success_count = 0
        error_count = 0
        errors = []
        processed_ids: List[Union[UUID, str]] = []

        for entity in entities:
            try:
                await self.save_model(entity)
                success_count += 1
                processed_ids.append(entity.id)
            except Exception as e:
                error_count += 1
                errors.append({"id": entity.id, "error": str(e)})

        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=success_count,
            error_count=error_count
        )

    async def bulk_update(self, entities: List[Model]) -> BulkOperationResult:
        """Пакетное обновление сущностей."""
        success_count = 0
        error_count = 0
        errors = []
        processed_ids: List[Union[UUID, str]] = []

        for entity in entities:
            try:
                await self.save_model(entity)
                success_count += 1
                processed_ids.append(entity.id)
            except Exception as e:
                error_count += 1
                errors.append({"id": entity.id, "error": str(e)})

        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=success_count,
            error_count=error_count
        )

    async def bulk_delete(self, entity_ids: List[Union[UUID, str]]) -> BulkOperationResult:
        """Пакетное удаление сущностей."""
        success_count = 0
        error_count = 0
        errors = []
        processed_ids: List[Union[UUID, str]] = []

        for entity_id in entity_ids:
            try:
                if await self.delete(entity_id):
                    success_count += 1
                    processed_ids.append(entity_id)
                else:
                    error_count += 1
                    errors.append({"id": entity_id, "error": "Entity not found"})
            except Exception as e:
                error_count += 1
                errors.append({"id": entity_id, "error": str(e)})

        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=success_count,
            error_count=error_count
        )

    async def bulk_upsert(self, entities: List[Model], conflict_fields: List[str]) -> BulkOperationResult:
        """Пакетное upsert сущностей."""
        success_count = 0
        error_count = 0
        errors = []
        processed_ids: List[Union[UUID, str]] = []

        for entity in entities:
            try:
                await self.save_model(entity)
                success_count += 1
                processed_ids.append(entity.id)
            except Exception as e:
                error_count += 1
                errors.append({"id": entity.id, "error": str(e)})

        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=success_count,
            error_count=error_count
        )

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Model]:
        """Получить из кэша."""
        cache_key = str(key)
        if cache_key in self.cache:
            ttl = self.cache_ttl.get(cache_key)
            if ttl and datetime.now() < ttl:
                return cast(Optional[Model], self.cache[cache_key])
            else:
                # Удаляем устаревшую запись
                del self.cache[cache_key]
                if cache_key in self.cache_ttl:
                    del self.cache_ttl[cache_key]
        return None

    async def set_cache(self, key: Union[UUID, str], entity: Model, ttl: Optional[int] = None) -> None:
        """Установить в кэш."""
        cache_key = str(key)
        self.cache[cache_key] = entity
        if ttl:
            self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl)

    async def _evict_cache(self) -> None:
        """Удалить устаревшие записи из кэша."""
        current_time = datetime.now()
        expired_keys = [
            key for key, ttl in self.cache_ttl.items() 
            if current_time > ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.cache_ttl[key]
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        if len(self.cache) > self.cache_max_size:
            # Удаляем самые старые записи
            sorted_keys = sorted(self.cache_ttl.keys(), key=lambda k: self.cache_ttl[k])
            keys_to_remove = sorted_keys[:len(self.cache) - self.cache_max_size]
            for key in keys_to_remove:
                del self.cache[key]
                del self.cache_ttl[key]
            self.logger.info(f"Cleaned up {len(keys_to_remove)} cache entries due to size limit")

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        cache_key = str(key)
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.cache_ttl:
            del self.cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        self.cache.clear()
        self.cache_ttl.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        stats = {
            "total_models": len(self.models),
            "total_predictions": len(self.predictions),
            "cache_size": len(self.cache),
            "cache_hit_rate": self._get_cache_hit_rate(),
            "last_cleanup": datetime.now().isoformat()
        }
        return RepositoryResponse(success=True, data=stats)

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        return {
            "total_trades": len(self.models),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": "0.0",
            "profit_factor": "0.0",
            "sharpe_ratio": "0.0",
            "max_drawdown": "0.0",
            "total_return": "0.0",
            "average_trade": "0.0",
            "calmar_ratio": "0.0",
            "sortino_ratio": "0.0",
            "var_95": "0.0",
            "cvar_95": "0.0"
        }

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        stats = {
            "cache_size": len(self.cache),
            "cache_hit_rate": self._get_cache_hit_rate(),
            "cache_miss_rate": 1.0 - self._get_cache_hit_rate(),
            "ttl_entries": len(self.cache_ttl)
        }
        return RepositoryResponse(success=True, data=stats)

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await self._evict_cache()
                await self._cleanup_cache()
                self.metrics["last_cleanup"] = datetime.now()
                await asyncio.sleep(60)  # Проверяем каждую минуту
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")
                await asyncio.sleep(60)

    def _matches_filter(self, model: Model, filter_item: QueryFilter) -> bool:
        """Проверить соответствие модели фильтру."""
        if filter_item.field == "model_type":
            return bool(str(model.model_type) == filter_item.value)
        elif filter_item.field == "status":
            return bool(model.status.value == filter_item.value)
        elif filter_item.field == "trading_pair":
            return bool(str(model.trading_pair) == filter_item.value)
        return True

    def _get_cache_hit_rate(self) -> float:
        """Получить процент попаданий в кэш."""
        cache_hits = int(str(self.metrics.get("cache_hits", 0) or 0))
        cache_misses = int(str(self.metrics.get("cache_misses", 0) or 0))
        total_requests = cache_hits + cache_misses
        
        if total_requests == 0:
            return 0.0
        
        return float(cache_hits) / float(total_requests)


class PostgresMLRepository(MLRepositoryProtocol):
    """
    PostgreSQL реализация репозитория ML моделей.
    """

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None) -> None:
        self.connection_string = connection_string
        self._pool: Optional[Any] = None
        self._cache_service = cache_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stats = {
            "total_operations": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def _get_pool(self) -> Any:
        """Получить пул соединений."""
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
        return self._pool

    async def _execute_with_retry(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Выполнить операцию с повторными попытками."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._stats["total_operations"] += 1
                return await operation(*args, **kwargs)
            except Exception as e:
                self._stats["errors"] += 1
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка

    async def save_model(self, model: Model) -> Model:
        """Сохранить модель."""
        async def _save_operation(conn: Any) -> Model:
            query = """
                INSERT INTO ml_models (id, name, model_type, status, trading_pair, 
                                     accuracy, created_at, updated_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    model_type = EXCLUDED.model_type,
                    status = EXCLUDED.status,
                    trading_pair = EXCLUDED.trading_pair,
                    accuracy = EXCLUDED.accuracy,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                model.id,
                model.name,
                model.model_type.value,
                model.status.value,
                str(model.trading_pair),
                float(model.accuracy) if hasattr(model, 'accuracy') else 0.0,
                model.created_at,
                model.updated_at,
                json.dumps(model.metadata) if hasattr(model, 'metadata') else '{}'
            )
            return self._row_to_model(row)
        
        pool = await self._get_pool()
        return cast(Model, await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_save_operation)
        ))

    async def get_model(self, model_id: Union[UUID, str]) -> Optional[Model]:
        async def _get_operation(conn: Any) -> Optional[Model]:
            row = await conn.fetchrow(
                "SELECT * FROM ml_models WHERE id = $1 AND deleted_at IS NULL",
                model_id
            )
            return self._row_to_model(row) if row else None
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_operation)
        )
        return result if isinstance(result, (Model, type(None))) else None

    async def get_models_by_type(self, model_type: str) -> List[Model]:
        async def _get_operation(conn: Any) -> List[Model]:
            rows = await conn.fetch(
                "SELECT * FROM ml_models WHERE model_type = $1 AND deleted_at IS NULL",
                model_type
            )
            return [self._row_to_model(row) for row in rows]
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_operation)
        )
        return result if isinstance(result, list) else []

    async def get_best_model(self, model_type: str) -> Optional[Model]:
        async def _get_operation(conn: Any) -> Optional[Model]:
            row = await conn.fetchrow(
                "SELECT * FROM ml_models WHERE model_type = $1 AND deleted_at IS NULL ORDER BY accuracy DESC LIMIT 1",
                model_type
            )
            return self._row_to_model(row) if row else None
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_operation)
        )
        return result if isinstance(result, (Model, type(None))) else None

    async def save(self, entity: Model) -> Model:
        """Сохранить сущность."""
        return await self.save_model(entity)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Model]:
        """Получить сущность по ID."""
        return await self.get_model(entity_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Model]:
        async def _get_all_operation(conn: Any) -> List[Model]:
            query = "SELECT * FROM ml_models WHERE deleted_at IS NULL"
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            rows = await conn.fetch(query)
            return [self._row_to_model(row) for row in rows]
        pool = await self._get_pool()
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_get_all_operation)
        )
        return result if isinstance(result, list) else []

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить модель."""
        async def _delete_operation(conn: Any) -> bool:
            result = await conn.execute("""
                DELETE FROM ml_models WHERE id = $1
            """, str(entity_id))
            return result == "DELETE 1"
        pool = await self._get_pool()
        result = await self._execute_with_retry(_delete_operation, pool)
        return bool(result)

    async def get_active_models(self) -> List[Model]:
        """Получить активные модели."""
        async def _get_operation(conn: Any) -> List[Model]:
            rows = await conn.fetch("""
                SELECT * FROM ml_models WHERE status = 'active' AND deleted_at IS NULL
            """)
            return [self._row_to_model(row) for row in rows]
        pool = await self._get_pool()
        result = await self._execute_with_retry(_get_operation, pool)
        return cast(List[Model], result)

    async def get_model_by_trading_pair(self, trading_pair: str) -> List[Model]:
        """Получить модели по торговой паре."""
        async def _get_operation(conn: Any) -> List[Model]:
            rows = await conn.fetch("""
                SELECT * FROM ml_models WHERE trading_pair = $1 AND deleted_at IS NULL
            """, trading_pair)
            return [self._row_to_model(row) for row in rows]
        pool = await self._get_pool()
        result = await self._execute_with_retry(_get_operation, pool)
        return cast(List[Model], result)

    async def save_prediction(self, prediction: Prediction) -> Prediction:
        """Сохранить предсказание."""
        async def _save_operation(conn: Any) -> Prediction:
            query = """
                INSERT INTO predictions (id, model_id, trading_pair, predicted_value, 
                                       confidence, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    predicted_value = EXCLUDED.predicted_value,
                    confidence = EXCLUDED.confidence,
                    timestamp = EXCLUDED.timestamp,
                    metadata = EXCLUDED.metadata
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                prediction.id,
                prediction.model_id,
                str(prediction.trading_pair),
                float(prediction.value) if prediction.value is not None else 0.0,
                float(prediction.confidence) if prediction.confidence is not None else 0.0,
                prediction.timestamp,
                json.dumps(prediction.metadata) if prediction.metadata else None
            )
            return self._row_to_prediction(row)
        pool = await self._get_pool()
        result = await self._execute_with_retry(_save_operation, pool)
        return cast(Prediction, result)

    async def get_predictions_by_model(self, model_id: UUID, limit: int = 100) -> List[Prediction]:
        """Получить предсказания модели."""
        async def _get_operation(conn: Any) -> List[Prediction]:
            rows = await conn.fetch("""
                SELECT * FROM predictions WHERE model_id = $1 ORDER BY timestamp DESC LIMIT $2
            """, str(model_id), limit)
            return [self._row_to_prediction(row) for row in rows]
        pool = await self._get_pool()
        result = await self._execute_with_retry(_get_operation, pool)
        return cast(List[Prediction], result)

    async def get_latest_prediction(self, model_id: UUID) -> Optional[Prediction]:
        """Получить последнее предсказание модели."""
        async def _get_operation(conn: Any) -> Optional[Prediction]:
            row = await conn.fetchrow("""
                SELECT * FROM predictions WHERE model_id = $1 ORDER BY timestamp DESC LIMIT 1
            """, str(model_id))
            return self._row_to_prediction(row) if row else None
        pool = await self._get_pool()
        result = await self._execute_with_retry(_get_operation, pool)
        return cast(Optional[Prediction], result)

    async def update_model_metrics(self, model_id: UUID, metrics: Dict[str, float]) -> bool:
        """Обновить метрики модели."""
        async def _update_operation(conn: Any) -> bool:
            query = """
                UPDATE ml_models 
                SET accuracy = $2, metadata = jsonb_set(metadata, '{metrics}', $3::jsonb), updated_at = NOW()
                WHERE id = $1
            """
            result = await conn.execute(
                query,
                str(model_id),
                metrics.get("accuracy", 0.0),
                json.dumps(metrics)
            )
            return result == "UPDATE 1"
        pool = await self._get_pool()
        result = await self._execute_with_retry(_update_operation, pool)
        return bool(result)

    async def update(self, entity: Model) -> Model:
        """Обновить сущность."""
        return await self.save_model(entity)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление сущности."""
        async def _soft_delete_operation(conn: Any) -> bool:
            result = await conn.execute("""
                UPDATE ml_models SET deleted_at = NOW() WHERE id = $1
            """, str(entity_id))
            return result == "UPDATE 1"
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(_soft_delete_operation, pool)
        return bool(result)

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить сущность."""
        async def _restore_operation(conn: Any) -> bool:
            result = await conn.execute("""
                UPDATE ml_models SET deleted_at = NULL WHERE id = $1
            """, str(entity_id))
            return result == "UPDATE 1"
        
        pool = await self._get_pool()
        result = await self._execute_with_retry(_restore_operation, pool)
        return bool(result)

    async def find_by(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> List[Model]:
        """Найти сущности по фильтрам."""
        async def _find_operation(conn: Any) -> List[Model]:
            where_clause, params = self._build_where_clause(filters)
            query = f"SELECT * FROM ml_models WHERE {where_clause} AND deleted_at IS NULL"
            
            if options and options.sort_orders:
                query += self._build_order_clause(options.sort_orders)
            
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            
            rows = await conn.fetch(query, *params)
            return [self._row_to_model(row) for row in rows]
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_find_operation)
        )
        return cast(List[Model], result if isinstance(result, list) else [])

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Model]:
        """Найти одну сущность по фильтрам."""
        models = await self.find_by(filters)
        return models[0] if models else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование сущности."""
        async def _exists_operation(conn: Any) -> bool:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_models WHERE id = $1 AND deleted_at IS NULL",
                entity_id
            )
            return result > 0
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_exists_operation)
        )
        return result

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество сущностей."""
        async def _count_operation(conn: Any) -> int:
            if filters:
                where_clause, params = self._build_where_clause(filters)
                query = f"SELECT COUNT(*) FROM ml_models WHERE {where_clause} AND deleted_at IS NULL"
                result = await conn.fetchval(query, *params)
            else:
                result = await conn.fetchval("SELECT COUNT(*) FROM ml_models WHERE deleted_at IS NULL")
            return result or 0
        
        pool = await self._get_pool()
        result: Any = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_count_operation)
        )
        return result if result is not None else 0

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[Model]:
        """Потоковое чтение сущностей."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = "SELECT * FROM ml_models WHERE deleted_at IS NULL"
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            rows = await conn.fetch(query)
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                for row in batch:
                    yield self._row_to_model(row)
                await asyncio.sleep(0)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                class PostgresMLTransaction(TransactionProtocol):
                    def __init__(self, conn: Any) -> None:
                        self.conn = conn
                        self._active = True
                    async def __aenter__(self) -> "PostgresMLTransaction":
                        return self
                    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                        self._active = False
                    async def commit(self) -> None:
                        pass
                    async def rollback(self) -> None:
                        self._active = False
                    async def is_active(self) -> bool:
                        return self._active
                yield PostgresMLTransaction(connection)

    async def execute_in_transaction(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции в транзакции."""
        async def _transaction_operation(conn: Any) -> Any:
            return await operation(*args, **kwargs)
        
        pool = await self._get_pool()
        return await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_transaction_operation)
        )

    async def bulk_save(self, entities: List[Model]) -> BulkOperationResult:
        """Пакетное сохранение сущностей."""
        success_count = 0
        error_count = 0
        errors = []
        processed_ids: List[Union[UUID, str]] = []

        for entity in entities:
            try:
                await self.save_model(entity)
                success_count += 1
                processed_ids.append(entity.id)
            except Exception as e:
                error_count += 1
                errors.append({"id": entity.id, "error": str(e)})

        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=success_count,
            error_count=error_count
        )

    async def bulk_update(self, entities: List[Model]) -> BulkOperationResult:
        """Пакетное обновление сущностей."""
        return await self.bulk_save(entities)

    async def bulk_delete(self, entity_ids: List[Union[UUID, str]]) -> BulkOperationResult:
        """Пакетное удаление сущностей."""
        success_count = 0
        error_count = 0
        errors = []
        processed_ids: List[Union[UUID, str]] = []

        for entity_id in entity_ids:
            try:
                if await self.delete(entity_id):
                    success_count += 1
                    processed_ids.append(entity_id)
                else:
                    error_count += 1
                    errors.append({"id": entity_id, "error": "Entity not found"})
            except Exception as e:
                error_count += 1
                errors.append({"id": entity_id, "error": str(e)})

        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=success_count,
            error_count=error_count
        )

    async def bulk_upsert(self, entities: List[Model], conflict_fields: List[str]) -> BulkOperationResult:
        """Пакетное upsert сущностей."""
        return await self.bulk_save(entities)

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Model]:
        """Получить из кэша."""
        if self._cache_service:
            result = await self._cache_service.get(str(key))
            return result if isinstance(result, Model) else None
        return None

    async def set_cache(self, key: Union[UUID, str], entity: Model, ttl: Optional[int] = None) -> None:
        """Установить кэш."""
        if self._cache_service:
            await self._cache_service.set(str(key), entity, ttl)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        if self._cache_service:
            await self._cache_service.delete(str(key))

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        if self._cache_service:
            await self._cache_service.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            total_models = await conn.fetchval("SELECT COUNT(*) FROM ml_models WHERE deleted_at IS NULL")
            total_predictions = await conn.fetchval("SELECT COUNT(*) FROM predictions")
            stats = {
                "total_models": total_models or 0,
                "total_predictions": total_predictions or 0,
                "cache_hit_rate": 0.0,
                "last_operation": datetime.now().isoformat() if isinstance(datetime.now(), datetime) else str(datetime.now())
            }
            return RepositoryResponse(success=True, data=stats)

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": "0.0",
            "profit_factor": "0.0",
            "sharpe_ratio": "0.0",
            "max_drawdown": "0.0",
            "total_return": "0.0",
            "average_trade": "0.0",
            "calmar_ratio": "0.0",
            "sortino_ratio": "0.0",
            "var_95": "0.0",
            "cvar_95": "0.0"
        }

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        if self._cache_service:
            stats = await self._cache_service.get_stats()
            return RepositoryResponse(success=True, data=stats)
        else:
            return RepositoryResponse(
                success=True,
                data={
                    "cache_size": 0,
                    "cache_hit_rate": 0.0,
                    "cache_miss_rate": 1.0
                }
            )

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat()
            }

    def _row_to_model(self, row: Any) -> Model:
        """Преобразовать строку БД в объект Model."""
        trading_pair = TradingPair(
            symbol=row["trading_pair"],
            base_currency=Currency("USD"),  # Заглушка
            quote_currency=Currency("USD")  # Заглушка
        )
        return Model(
            id=row["id"],
            name=row["name"],
            model_type=ModelType(row["model_type"]),
            status=ModelStatus(row["status"]),
            trading_pair=str(trading_pair),
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def _row_to_prediction(self, row: Any) -> Prediction:
        """Преобразовать строку БД в объект Prediction."""
        trading_pair = TradingPair(
            symbol=row["trading_pair"],
            base_currency=Currency("USD"),  # Заглушка
            quote_currency=Currency("USD")  # Заглушка
        )
        return Prediction(
            id=row["id"],
            model_id=row["model_id"],
            trading_pair=str(trading_pair),
            value=row.get("predicted_value", 0.0),
            confidence=row.get("confidence", 0.0),
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if "metadata" in row and row["metadata"] else {}
        )

    def _build_where_clause(self, filters: List[QueryFilter]) -> tuple[str, List[Any]]:
        """Построить WHERE условие для фильтров."""
        conditions = []
        params = []
        param_count = 0
        
        for filter_item in filters:
            param_count += 1
            if filter_item.operator == QueryOperator.EQUALS:
                conditions.append(f"{filter_item.field} = ${param_count}")
                params.append(filter_item.value)
            elif filter_item.operator == QueryOperator.NOT_EQUALS:
                conditions.append(f"{filter_item.field} != ${param_count}")
                params.append(filter_item.value)
            elif filter_item.operator == QueryOperator.GREATER_THAN:
                conditions.append(f"{filter_item.field} > ${param_count}")
                params.append(filter_item.value)
            elif filter_item.operator == QueryOperator.LESS_THAN:
                conditions.append(f"{filter_item.field} < ${param_count}")
                params.append(filter_item.value)
            elif filter_item.operator == QueryOperator.LIKE:
                conditions.append(f"{filter_item.field} ILIKE ${param_count}")
                params.append(f"%{filter_item.value}%")
            elif filter_item.operator == QueryOperator.IN:
                conditions.append(f"{filter_item.field} = ANY(${param_count})")
                params.append(filter_item.value)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params

    def _build_order_clause(self, sort_orders: List[Any]) -> str:
        """Построить ORDER BY условие."""
        if not sort_orders:
            return ""
        
        order_parts = []
        for sort_order in sort_orders:
            direction = "DESC" if sort_order.direction == "desc" else "ASC"
            order_parts.append(f"{sort_order.field} {direction}")
        
        return f" ORDER BY {', '.join(order_parts)}"

    async def close(self) -> None:
        """Закрыть соединения."""
        if self._pool:
            await self._pool.close()
