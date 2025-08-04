"""
Реализация репозитория стратегий.
"""

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, Union, cast, AsyncContextManager
from uuid import UUID

import asyncpg
from sqlalchemy import select
import json
import time
from contextlib import AbstractAsyncContextManager, _AsyncGeneratorContextManager

from domain.entities.strategy import (
    Strategy,
    StrategyStatus,
    StrategyType,
)
from domain.exceptions.protocol_exceptions import (
    EntityNotFoundError,
    EntitySaveError,
    EntityUpdateError,
    TransactionError,
    ValidationError,
)
from domain.exceptions.base_exceptions import RepositoryError
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    QueryOptions,
    RepositoryResponse,
    RepositoryState,
    StrategyRepositoryProtocol,
    PerformanceMetricsDict,
    HealthCheckDict,
    TransactionProtocol,
)
from domain.type_definitions.repository_types import QueryFilter
from domain.type_definitions import ModelId, PredictionId, StrategyId
from domain.type_definitions.repository_types import (
    PerformanceMetrics,
    HealthCheckResult,
    CacheKey,
)


class InMemoryStrategyRepository(StrategyRepositoryProtocol):
    """
    Сверхпродвинутая in-memory реализация репозитория стратегий.
    - Кэширование с TTL
    - Асинхронные транзакции
    - Индексация по типу, статусу, владельцу
    - Хранение истории performance
    - Мониторинг и health-check
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._strategies: Dict[Union[UUID, str], Strategy] = {}
        self._strategies_by_type: Dict[str, List[str]] = defaultdict(list)
        self._strategies_by_status: Dict[str, List[str]] = defaultdict(list)
        self._performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._cache: Dict[Union[UUID, str], Strategy] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._last_cleanup = datetime.now()
        self._state = RepositoryState.CONNECTED
        self._startup_time = datetime.now()
        self.logger.info("InMemoryStrategyRepository initialized")

    # CRUD операции
    async def save_strategy(self, strategy: Strategy) -> Strategy:
        """Сохранить стратегию."""
        try:
            sid = str(strategy.id)
            self._strategies[sid] = strategy

            # Обновление индексов
            strategy_type = strategy.strategy_type.value
            if sid not in self._strategies_by_type[strategy_type]:
                self._strategies_by_type[strategy_type].append(sid)
            strategy_status = strategy.status.value
            if sid not in self._strategies_by_status[strategy_status]:
                self._strategies_by_status[strategy_status].append(sid)

            await self.invalidate_cache(sid)
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to save strategy {strategy.id}: {e}")
            raise

    async def get_strategy(self, strategy_id: Union[UUID, str]) -> Optional[Strategy]:
        sid = str(strategy_id)
        cache_key = f"strategy:{sid}"
        cached = await self.get_from_cache(cache_key)
        if cached:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        strategy = self._strategies.get(sid)
        if strategy:
            await self.set_cache(cache_key, strategy)
        return strategy

    async def get_strategies_by_type(self, strategy_type: str) -> List[Strategy]:
        ids = self._strategies_by_type.get(strategy_type, [])
        return [self._strategies[sid] for sid in ids if sid in self._strategies]

    async def get_active_strategies(self) -> List[Strategy]:
        ids = self._strategies_by_status.get(StrategyStatus.ACTIVE.value, [])
        return [self._strategies[sid] for sid in ids if sid in self._strategies]

    async def update_strategy_performance(
        self, strategy_id: StrategyId, performance_metrics: Dict[str, Any]
    ) -> bool:
        sid = str(strategy_id)
        self._performance_history[sid].append(
            {"timestamp": datetime.now(), "metrics": performance_metrics}
        )
        return True

    async def get_strategy_performance_history(
        self,
        strategy_id: StrategyId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        sid = str(strategy_id)
        history = self._performance_history.get(sid, [])
        if start_date:
            history = [h for h in history if h["timestamp"] >= start_date]
        if end_date:
            history = [h for h in history if h["timestamp"] <= end_date]
        return history

    # Кэширование
    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Strategy]:
        cache_key = str(key)
        if cache_key not in self._cache:
            return None
        if cache_key in self._cache_ttl and datetime.now() > self._cache_ttl[cache_key]:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
            return None
        return self._cache[cache_key]

    async def set_cache(
        self, key: Union[UUID, str], entity: Strategy, ttl: Optional[int] = None
    ) -> None:
        cache_key = str(key)
        ttl_seconds = ttl or self._cache_ttl_seconds
        if len(self._cache) >= self._cache_max_size:
            await self._evict_cache()
        self._cache[cache_key] = entity
        self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        cache_key = str(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_ttl:
            del self._cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_ttl.clear()

    async def _evict_cache(self) -> None:
        """Вытеснить старые записи из кэша."""
        if not self._cache:
            return
        oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
        await self.invalidate_cache(oldest_key)

    # Транзакции
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Транзакция."""
        class StrategyTransaction(TransactionProtocol):
            def __init__(self, repo: "InMemoryStrategyRepository") -> None:
                self.repo = repo
                self.active = True

            async def __aenter__(self) -> "StrategyTransaction":
                return self

            async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
                if exc_type is not None:
                    await self.rollback()

            async def commit(self) -> None:
                self.active = False

            async def rollback(self) -> None:
                self.active = False

            async def is_active(self) -> bool:
                return self.active

        transaction = StrategyTransaction(self)
        yield transaction

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[Strategy]:
        """Потоковая обработка стратегий."""
        strategies = list(self._strategies.values())
        if options and options.filters:
            strategies = self._apply_filters(strategies, options.filters)
        if options and hasattr(options, 'sort_orders') and options.sort_orders:
            strategies = self._apply_sort(strategies, options.sort_orders)
        if options and options.pagination:
            strategies = self._apply_pagination(strategies, options.pagination)
        
        for i in range(0, len(strategies), batch_size):
            batch = strategies[i:i + batch_size]
            for strategy in batch:
                yield strategy

    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[Strategy]:
        """Поиск стратегий по фильтрам."""
        strategies = list(self._strategies.values())
        if filters:
            strategies = self._apply_filters(strategies, filters)
        if options and hasattr(options, 'sort_orders') and options.sort_orders:
            strategies = self._apply_sort(strategies, options.sort_orders)
        if options and options.pagination:
            strategies = self._apply_pagination(strategies, options.pagination)
        return strategies

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Strategy]:
        """Поиск одной стратегии по фильтрам."""
        strategies = list(self._strategies.values())
        filtered = self._apply_filters(strategies, filters)
        return filtered[0] if filtered else None

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчет количества стратегий."""
        strategies = list(self._strategies.values())
        if filters:
            strategies = self._apply_filters(strategies, filters)
        return len(strategies)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Strategy]:
        """Получить стратегию по ID."""
        return await self.get_strategy(entity_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Strategy]:
        """Получить все стратегии."""
        return list(self._strategies.values())

    async def update(self, entity: Strategy) -> Strategy:
        """Обновить стратегию."""
        return await self.save_strategy(entity)

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить стратегию."""
        sid = str(entity_id)
        if sid not in self._strategies:
            return False
        
        strategy = self._strategies[sid]
        
        # Удаление из индексов
        strategy_type = strategy.strategy_type.value
        if sid in self._strategies_by_type[strategy_type]:
            self._strategies_by_type[strategy_type].remove(sid)
        
        strategy_status = strategy.status.value
        if sid in self._strategies_by_status[strategy_status]:
            self._strategies_by_status[strategy_status].remove(sid)
        
        # Удаление из основного хранилища
        del self._strategies[sid]
        
        # Инвалидация кэша
        await self.invalidate_cache(sid)
        
        return True

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление стратегии."""
        strategy = await self.get_strategy(entity_id)
        if not strategy:
            return False
        strategy.status = StrategyStatus.INACTIVE
        await self.save_strategy(strategy)
        return True

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить стратегию."""
        strategy = await self.get_strategy(entity_id)
        if not strategy:
            return False
        strategy.status = StrategyStatus.ACTIVE
        await self.save_strategy(strategy)
        return True

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        return RepositoryResponse(
            success=True,
            data={
                "total_entities": len(self._strategies),
                "cache_size": len(self._cache),
                "cache_hit_rate": self._get_cache_hit_rate(),
                "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
            },
            total_count=len(self._strategies),
        )

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
            "cvar_95": "0.0",
        }

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "response_time": 0.0,
            "error_count": 0,
            "last_error": None,
            "uptime": (datetime.now() - self._startup_time).total_seconds(),
        }

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(60)  # Каждую минуту
                current_time = datetime.now()
                expired_keys = [
                    key for key, ttl in self._cache_ttl.items()
                    if current_time > ttl
                ]
                for key in expired_keys:
                    await self.invalidate_cache(key)
                    self._cache_evictions += 1
                self._last_cleanup = datetime.now()
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")

    async def save(self, entity: Strategy) -> Strategy:
        """Сохранить стратегию."""
        return await self.save_strategy(entity)

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        return RepositoryResponse(
            success=True,
            data={
                "cache_size": len(self._cache),
                "cache_hit_rate": self._get_cache_hit_rate(),
                "cache_ttl_seconds": self._cache_ttl_seconds,
                "cache_max_size": self._cache_max_size,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_evictions": self._cache_evictions,
            },
        )

    def _apply_filters(self, strategies: List[Strategy], filters: List[QueryFilter]) -> List[Strategy]:
        """Применить фильтры к стратегиям."""
        filtered_strategies = strategies
        for filter_obj in filters:
            if filter_obj.field == "strategy_type":
                filtered_strategies = [
                    s for s in filtered_strategies
                    if s.strategy_type.value == str(filter_obj.value)
                ]
            elif filter_obj.field == "status":
                filtered_strategies = [
                    s for s in filtered_strategies
                    if s.status.value == str(filter_obj.value)
                ]
            elif filter_obj.field == "name":
                filtered_strategies = [
                    s for s in filtered_strategies
                    if filter_obj.value.lower() in s.name.lower()
                ]
        return filtered_strategies

    def _apply_sort(self, strategies: List[Strategy], sort_orders: List[Any]) -> List[Strategy]:
        """Применить сортировку к стратегиям."""
        for sort_order in sort_orders:
            if sort_order.field == "name":
                reverse = sort_order.direction == "desc"
                strategies.sort(key=lambda s: s.name, reverse=reverse)
            elif sort_order.field == "created_at":
                reverse = sort_order.direction == "desc"
                strategies.sort(key=lambda s: s.created_at, reverse=reverse)
        return strategies

    def _apply_pagination(self, strategies: List[Strategy], pagination: Any) -> List[Strategy]:
        """Применить пагинацию к стратегиям."""
        if pagination.offset is not None and pagination.limit is not None:
            return strategies[pagination.offset:pagination.offset + pagination.limit]
        elif pagination.page and pagination.page_size:
            start = (pagination.page - 1) * pagination.page_size
            end = start + pagination.page_size
            return strategies[start:end]
        return strategies

    def _get_cache_hit_rate(self) -> float:
        """Получить hit rate кэша."""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests == 0:
            return 0.0
        return self._cache_hits / total_requests


class PostgresStrategyRepository(StrategyRepositoryProtocol):
    """
    PostgreSQL реализация репозитория стратегий.
    """

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None) -> None:
        self.connection_string = connection_string
        self._pool: Optional[asyncpg.Pool] = None
        self._cache_service = cache_service
        self._cache: Dict[Union[UUID, str], Strategy] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._max_cache_size = 1000
        self._cache_ttl_seconds = 300
        self._stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "last_operation_time": 0.0,
        }
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._is_initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._strategies: Dict[Union[UUID, str], Strategy] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        self._state = RepositoryState.CONNECTED
        self._startup_time = datetime.now()
        asyncio.create_task(self._initialize_pool())
        self.logger.info("PostgresStrategyRepository initialized")

    async def _get_pool(self) -> asyncpg.Pool:
        if not self._pool:
            await self._initialize_pool()
        return self._pool

    async def _execute_with_retry(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)
            except asyncpg.ConnectionDoesNotExistError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
                await self._initialize_pool()

    async def save_strategy(self, strategy: Strategy) -> Strategy:
        """Сохранить стратегию."""
        async def _save_operation(conn: Any) -> None:
            await conn.execute("""
                INSERT INTO strategies (
                    id, name, type, status, config, performance_metrics,
                    created_at, updated_at, is_active, risk_level,
                    target_profit, stop_loss, max_position_size,
                    allowed_pairs, trading_hours, volatility_threshold
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    status = EXCLUDED.status,
                    config = EXCLUDED.config,
                    performance_metrics = EXCLUDED.performance_metrics,
                    updated_at = EXCLUDED.updated_at,
                    is_active = EXCLUDED.is_active,
                    risk_level = EXCLUDED.risk_level,
                    target_profit = EXCLUDED.target_profit,
                    stop_loss = EXCLUDED.stop_loss,
                    max_position_size = EXCLUDED.max_position_size,
                    allowed_pairs = EXCLUDED.allowed_pairs,
                    trading_hours = EXCLUDED.trading_hours,
                    volatility_threshold = EXCLUDED.volatility_threshold
            """, strategy.id, strategy.name, strategy.strategy_type.value, strategy.status.value,
                 json.dumps(strategy.parameters.to_dict()), json.dumps(strategy.performance.to_dict()),
                 strategy.created_at, strategy.updated_at, strategy.is_active,
                 strategy.parameters.get_parameter("risk_level", 0.02), strategy.parameters.get_parameter("target_profit", 0.05), strategy.parameters.get_parameter("stop_loss", 0.02),
                 strategy.parameters.get_parameter("max_position_size", 0.1), json.dumps(strategy.trading_pairs),
                 json.dumps(strategy.trading_pairs), strategy.parameters.get_parameter("volatility_threshold", 0.1))

        pool = await self._get_pool()
        await self._execute_with_retry(_save_operation, pool)
        await self.set_cache(strategy.id, strategy)
        return strategy

    async def get_strategy(self, strategy_id: Union[UUID, str]) -> Optional[Strategy]:
        """Получить стратегию по ID."""
        async def _get_operation(conn: Any) -> Any:
            row = await conn.fetchrow("""
                SELECT * FROM strategies WHERE id = $1 AND deleted_at IS NULL
            """, str(strategy_id))
            return row

        pool = await self._get_pool()
        row = await self._execute_with_retry(_get_operation, pool)
        if row:
            return self._row_to_strategy(row)
        return None

    async def get_strategies_by_type(self, strategy_type: str) -> List[Strategy]:
        """Получить стратегии по типу."""
        async def _get_operation(conn: Any) -> List[Any]:
            rows = await conn.fetch("""
                SELECT * FROM strategies 
                WHERE type = $1 AND deleted_at IS NULL
                ORDER BY created_at DESC
            """, strategy_type)
            return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

        pool = await self._get_pool()
        rows = await self._execute_with_retry(_get_operation, pool)
        return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

    async def get_active_strategies(self) -> List[Strategy]:
        """Получить активные стратегии."""
        async def _get_operation(conn: Any) -> List[Any]:
            rows = await conn.fetch("""
                SELECT * FROM strategies 
                WHERE is_active = true AND deleted_at IS NULL
                ORDER BY created_at DESC
            """)
            return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

        pool = await self._get_pool()
        rows = await self._execute_with_retry(_get_operation, pool)
        return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

    async def update_strategy_performance(self, strategy_id: StrategyId, performance_metrics: Dict[str, Any]) -> bool:
        """Обновить метрики производительности стратегии."""
        async def _update_operation(conn: Any) -> None:
            await conn.execute("""
                UPDATE strategies 
                SET performance_metrics = $2, updated_at = NOW()
                WHERE id = $1 AND deleted_at IS NULL
            """, str(strategy_id), json.dumps(performance_metrics))

        pool = await self._get_pool()
        await self._execute_with_retry(_update_operation, pool)
        await self.invalidate_cache(strategy_id)
        return True

    async def get_strategy_performance_history(
        self, strategy_id: StrategyId, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Получить историю производительности стратегии."""
        async def _get_operation(conn: Any) -> List[Any]:
            query = """
                SELECT performance_metrics, created_at 
                FROM strategy_performance_history 
                WHERE strategy_id = $1
            """
            params: List[Any] = [str(strategy_id)]
            
            if start_date:
                query += " AND created_at >= $2"
                params.append(start_date)
            if end_date:
                query += " AND created_at <= $3"
                params.append(end_date)
            
            query += " ORDER BY created_at DESC"
            rows = await conn.fetch(query, *params)
            return [{"metrics": json.loads(row["performance_metrics"]), "date": row["created_at"]} for row in rows]

        pool = await self._get_pool()
        rows = await self._execute_with_retry(_get_operation, pool)
        return [{"metrics": json.loads(row["performance_metrics"]), "date": row["created_at"]} for row in rows]

    async def save(self, entity: Strategy) -> Strategy:
        """Сохранить сущность."""
        return await self.save_strategy(entity)

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Strategy]:
        """Получить сущность по ID."""
        cached = await self.get_from_cache(entity_id)
        if cached:
            return cached

        strategy = await self.get_strategy(entity_id)
        if strategy:
            await self.set_cache(entity_id, strategy)
        return strategy

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Strategy]:
        """Получить все стратегии."""
        async def _get_operation(conn: Any) -> List[Any]:
            query = "SELECT * FROM strategies WHERE deleted_at IS NULL"
            params: List[Any] = []
            
            if options and options.filters:
                # Применяем фильтры
                for i, filter_item in enumerate(options.filters, start=len(params) + 1):
                    query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                    params.append(filter_item.value)
            
            if options and hasattr(options, 'sort_orders') and options.sort_orders:
                # Применяем сортировку
                sort_clauses = []
                for sort_order in options.sort_orders:
                    direction = "DESC" if sort_order.direction == "desc" else "ASC"
                    sort_clauses.append(f"{sort_order.field} {direction}")
                query += f" ORDER BY {', '.join(sort_clauses)}"
            else:
                query += " ORDER BY created_at DESC"
            
            if options and options.pagination:
                # Применяем пагинацию
                offset = (options.pagination.page - 1) * options.pagination.page_size
                query += f" LIMIT {options.pagination.page_size} OFFSET {offset}"
            
            rows = await conn.fetch(query, *params)
            return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

        pool = await self._get_pool()
        rows = await self._execute_with_retry(_get_operation, pool)
        return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить стратегию."""
        async def _delete_operation(conn: Any) -> bool:
            result = await conn.execute("""
                DELETE FROM strategies WHERE id = $1
            """, str(entity_id))
            return result == "DELETE 1"
        pool = await self._get_pool()
        success = await self._execute_with_retry(_delete_operation, pool)
        if success:
            await self.invalidate_cache(entity_id)
        return bool(success)  # type: ignore[no-any-return]

    async def update(self, entity: Strategy) -> Strategy:
        """Обновить стратегию."""
        return await self.save_strategy(entity)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """Мягкое удаление стратегии."""
        async def _soft_delete_operation(conn: Any) -> bool:
            result = await conn.execute("""
                UPDATE strategies 
                SET deleted_at = NOW(), is_active = false
                WHERE id = $1 AND deleted_at IS NULL
            """, str(entity_id))
            return result == "UPDATE 1"
        pool = await self._get_pool()
        success = await self._execute_with_retry(_soft_delete_operation, pool)
        if success:
            await self.invalidate_cache(entity_id)
        return bool(success)  # type: ignore[no-any-return]

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """Восстановить стратегию."""
        async def _restore_operation(conn: Any) -> bool:
            result = await conn.execute("""
                UPDATE strategies 
                SET deleted_at = NULL, is_active = true
                WHERE id = $1 AND deleted_at IS NOT NULL
            """, str(entity_id))
            return result == "UPDATE 1"
        pool = await self._get_pool()
        success = await self._execute_with_retry(_restore_operation, pool)
        if success:
            await self.invalidate_cache(entity_id)
        return bool(success)  # type: ignore[no-any-return]

    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[Strategy]:
        """Найти стратегии по фильтрам."""
        return await self._find_by_operation(await self._get_pool(), filters, options)

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Strategy]:
        """Найти одну стратегию по фильтрам."""
        strategies = await self.find_by(filters)
        return strategies[0] if strategies else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """Проверить существование стратегии."""
        return await self._exists_operation(await self._get_pool(), entity_id)

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество стратегий."""
        return await self._count_operation(await self._get_pool(), filters)

    async def stream(self, options: Optional[QueryOptions] = None, batch_size: int = 100) -> AsyncIterator[Strategy]:
        """Потоковая обработка стратегий."""
        async def _stream_operation(conn: Any) -> AsyncIterator[Strategy]:
            query = "SELECT * FROM strategies WHERE deleted_at IS NULL"
            params: List[Any] = []
            
            if options and options.filters:
                for i, filter_item in enumerate(options.filters, start=len(params) + 1):
                    query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                    params.append(filter_item.value)
            
            if options and hasattr(options, 'sort_orders') and options.sort_orders:
                sort_clauses = []
                for sort_order in options.sort_orders:
                    direction = "DESC" if sort_order.direction == "desc" else "ASC"
                    sort_clauses.append(f"{sort_order.field} {direction}")
                query += f" ORDER BY {', '.join(sort_clauses)}"
            else:
                query += " ORDER BY created_at DESC"
            
            async with conn.transaction():
                async for record in conn.cursor(query, *params):
                    yield self._row_to_strategy(record)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async for strategy in _stream_operation(conn):
                yield strategy

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            async with connection.transaction() as transaction:
                yield transaction

    async def execute_in_transaction(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Выполнить операцию в транзакции."""
        async with self.transaction() as transaction:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[Strategy]) -> BulkOperationResult:
        """Пакетное сохранение стратегий."""
        async def _bulk_save_operation(conn: Any) -> None:
            async with conn.transaction():
                for strategy in entities:
                    await conn.execute("""
                        INSERT INTO strategies (id, name, strategy_type, status, config, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            strategy_type = EXCLUDED.strategy_type,
                            status = EXCLUDED.status,
                            config = EXCLUDED.config,
                            updated_at = EXCLUDED.updated_at
                    """, str(strategy.id), strategy.name, strategy.strategy_type.value,
                         strategy.status.value, strategy.metadata, strategy.created_at, strategy.updated_at)
        
        pool = await self._get_pool()
        await self._execute_with_retry(_bulk_save_operation, pool.acquire())
        return BulkOperationResult(
            success_count=len(entities),
            error_count=0,
            errors=[],
            processed_ids=[str(s.id) for s in entities],
        )

    async def bulk_update(self, entities: List[Strategy]) -> BulkOperationResult:
        """Пакетное обновление стратегий."""
        return await self.bulk_save(entities)

    async def bulk_delete(self, entity_ids: List[Union[UUID, str]]) -> BulkOperationResult:
        """Пакетное удаление стратегий."""
        async def _bulk_delete_operation(conn: Any) -> None:
            async with conn.transaction():
                for entity_id in entity_ids:
                    await conn.execute(
                        "DELETE FROM strategies WHERE id = $1",
                        str(entity_id)
                    )
        
        pool = await self._get_pool()
        await self._execute_with_retry(_bulk_delete_operation, pool.acquire())
        return BulkOperationResult(
            success_count=len(entity_ids),
            error_count=0,
            errors=[],
            processed_ids=[str(eid) for eid in entity_ids],
        )

    async def bulk_upsert(self, entities: List[Strategy], conflict_fields: List[str]) -> BulkOperationResult:
        """Пакетное upsert стратегий."""
        return await self.bulk_save(entities)

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Strategy]:
        """Получить из кэша."""
        key_str = str(key)
        if key_str not in self._cache:
            return None
        if key_str in self._cache_ttl and datetime.now() > self._cache_ttl[key_str]:
            del self._cache[key_str]
            del self._cache_ttl[key_str]
            return None
        return self._cache[key_str]

    async def set_cache(self, key: Union[UUID, str], entity: Strategy, ttl: Optional[int] = None) -> None:
        """Установить в кэш."""
        key_str = str(key)
        self._cache[key_str] = entity
        ttl_seconds = ttl or self._cache_ttl_seconds
        self._cache_ttl[key_str] = datetime.now() + timedelta(seconds=int(ttl_seconds))
        await self._evict_cache()

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        key_str = str(key)
        if key_str in self._cache:
            del self._cache[key_str]
        if key_str in self._cache_ttl:
            del self._cache_ttl[key_str]

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._cache_ttl.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            total_count = await conn.fetchval("SELECT COUNT(*) FROM strategies")
            uptime_seconds = (datetime.now().timestamp() - self._startup_time.timestamp())
            return RepositoryResponse(
                success=True,
                data={
                    "total_entities": total_count or 0,
                    "cache_size": len(self._cache),
                    "cache_hit_rate": self._get_cache_hit_rate(),
                    "uptime_seconds": uptime_seconds,
                },
                total_count=total_count or 0,
            )

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        async def _performance_metrics_operation(conn: Any) -> PerformanceMetricsDict:
            # Получаем статистику из БД
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_entities,
                    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_query_time
                FROM strategies 
                WHERE deleted_at IS NULL
            """)
            
            return {
                "total_entities": stats["total_entities"] if stats else 0,
                "cache_hit_rate": self._get_cache_hit_rate(),
                "avg_query_time": float(stats["avg_query_time"]) if stats and stats["avg_query_time"] else 0.0,
                "error_rate": self._get_error_rate(),
                "last_cleanup": datetime.now().isoformat()
            }

        pool = await self._get_pool()
        result = await self._execute_with_retry(_performance_metrics_operation, pool)
        return result  # type: ignore[no-any-return]

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        async def _health_check_operation(conn: Any) -> HealthCheckDict:
            # Проверяем соединение с БД
            await conn.execute("SELECT 1")
            
            return {
                "status": "healthy",
                "database_connected": True,
                "cache_working": len(self._cache) >= 0,
                "last_check": datetime.now().isoformat(),
                "errors": []
            }

        try:
            pool = await self._get_pool()
            result: HealthCheckDict = await self._execute_with_retry(_health_check_operation, pool)
            return result  # type: ignore[no-any-return]
        except Exception as e:
            # Возвращаем результат проверки здоровья при ошибке
            return {
                "status": "unhealthy",
                "database_connected": False,
                "cache_working": False,
                "last_check": datetime.now().isoformat(),
                "errors": [str(e)]
            }  # type: ignore[no-any-return]

    def _row_to_strategy(self, row: Any) -> Strategy:
        """Преобразовать строку БД в объект Strategy."""
        return Strategy(
            id=row["id"],
            name=row["name"],
            strategy_type=row["type"],
            status=row["status"],
            metadata=json.loads(row["config"]) if row["config"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=row["is_active"],
            trading_pairs=json.loads(row["allowed_pairs"]) if row["allowed_pairs"] else []
        )

    async def _evict_cache(self) -> None:
        """Удалить устаревшие записи из кэша."""
        current_time = time.time()
        expired_keys = [
            key for key, ttl in self._cache_ttl.items()
            if current_time - ttl > self._cache_ttl_seconds  # type: ignore[operator]
        ]
        for key in expired_keys:
            await self.invalidate_cache(key)

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        if len(self._cache) > self._max_cache_size:
            # Удаляем самые старые записи
            sorted_keys = sorted(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
            keys_to_remove = sorted_keys[:len(self._cache) - self._max_cache_size]
            for key in keys_to_remove:
                await self.invalidate_cache(key)

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(60)  # Каждую минуту
                await self._evict_cache()
                await self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")

    async def close(self) -> None:
        """Закрыть репозиторий."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        if self._pool:
            await self._pool.close()

    async def _initialize_pool(self) -> None:
        """Инициализировать пул соединений."""
        if not self._is_initialized:
            self._pool = await asyncpg.create_pool(self.connection_string)
            self._is_initialized = True
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

    def _get_cache_hit_rate(self) -> float:
        """Получить процент попаданий в кэш."""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        return self._stats["cache_hits"] / total if total > 0 else 0.0

    def _get_error_rate(self) -> float:
        """Получить процент ошибок."""
        total = self._stats["total_operations"]
        return self._stats["errors"] / total if total > 0 else 0.0

    async def _find_by_operation(self, conn: Any, filters: List[QueryFilter], options: Optional[QueryOptions]) -> List[Strategy]:
        """Операция для поиска по фильтрам."""
        query = "SELECT * FROM strategies WHERE deleted_at IS NULL"
        params: List[Any] = []
        
        if filters:
            for i, filter_item in enumerate(filters, start=len(params) + 1):
                query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                params.append(filter_item.value)
        
        if options and hasattr(options, 'sort_orders') and options.sort_orders:
            sort_clauses = []
            for sort_order in options.sort_orders:
                direction = "DESC" if sort_order.direction == "desc" else "ASC"
                sort_clauses.append(f"{sort_order.field} {direction}")
            query += f" ORDER BY {', '.join(sort_clauses)}"
        else:
            query += " ORDER BY created_at DESC"
        
        if options and options.pagination:
            offset = (options.pagination.page - 1) * options.pagination.page_size
            query += f" LIMIT {options.pagination.page_size} OFFSET {offset}"
        
        rows = await conn.fetch(query, *params)
        return cast(List[Strategy], [self._row_to_strategy(row) for row in rows])

    async def _exists_operation(self, conn: Any, entity_id: Union[UUID, str]) -> bool:
        """Операция для проверки существования."""
        row = await conn.fetchrow("""
            SELECT 1 FROM strategies WHERE id = $1 AND deleted_at IS NULL
        """, str(entity_id))
        return row is not None

    async def _count_operation(self, conn: Any, filters: Optional[List[QueryFilter]]) -> int:
        """Операция для подсчета."""
        query = "SELECT COUNT(*) FROM strategies WHERE deleted_at IS NULL"
        params: List[Any] = []
        
        if filters:
            for i, filter_item in enumerate(filters, start=len(params) + 1):
                query += f" AND {filter_item.field} {filter_item.operator} ${i}"
                params.append(filter_item.value)
        
        result = await conn.fetchval(query, *params)
        return int(result) if result is not None else 0
