"""
Репозиторий торговых пар - промышленная реализация.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable
from uuid import UUID

import asyncpg
from sqlalchemy import select

from domain.entities.trading_pair import TradingPair
from domain.exceptions.protocol_exceptions import (
    EntityNotFoundError,
    EntitySaveError,
    EntityUpdateError,
    TransactionError,
    ValidationError,
)
from domain.repositories.base_repository import BaseRepository
from domain.value_objects import Currency
from domain.types import Symbol
from domain.types.repository_types import (
    BulkOperationResult,
    QueryFilter,
    QueryOptions,
    RepositoryResponse,
    PerformanceMetrics,
    HealthCheckResult,
    CacheKey,
    RepositoryState as RepositoryStateTypes,
)
from domain.protocols.repository_protocol import RepositoryState as RepositoryStateProtocol
from domain.protocols.repository_protocol import TradingPairRepositoryProtocol, TransactionProtocol
from domain.types.protocol_types import PerformanceMetricsDict, HealthCheckDict
from infrastructure.repositories.trading.models import TradingPairModel


class InMemoryTradingPairRepository(BaseRepository[TradingPair], TradingPairRepositoryProtocol):
    """
    In-memory реализация репозитория торговых пар.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._trading_pairs: Dict[str, TradingPair] = {}
        # Приводим типы кэша к совместимым с базовым классом
        self._cache: Dict[CacheKey, TradingPair] = {}
        self._cache_ttl: Dict[CacheKey, datetime] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        self._state = RepositoryState.CONNECTED
        self._startup_time = datetime.now()
        asyncio.create_task(self._background_cleanup())
        self.logger.info("InMemoryTradingPairRepository initialized")

    async def save(self, trading_pair: TradingPair) -> TradingPair:
        """Сохранить торговую пару."""
        pair_id = str(trading_pair.symbol)
        self._trading_pairs[pair_id] = trading_pair
        # Инвалидация кэша
        await self.invalidate_cache(CacheKey(pair_id))
        return trading_pair

    async def get_by_id(self, pair_id: Union[UUID, str]) -> Optional[TradingPair]:
        """Получить торговую пару по ID."""
        pair_id_str = str(pair_id)
        return self._trading_pairs.get(pair_id_str)

    async def get_all(
        self, options: Optional[QueryOptions] = None
    ) -> List[TradingPair]:
        """Получить все торговые пары."""
        pairs = list(self._trading_pairs.values())
        if options and options.filters:
            # Приводим типы фильтров к совместимым
            filters = [QueryFilter(f.field, f.operator, f.value) for f in options.filters]
            pairs = self._apply_filters(pairs, filters)
        if options and hasattr(options, 'sort_orders') and options.sort_orders:
            pairs = self._apply_sort(pairs, options.sort_orders)
        if options and options.pagination:
            pairs = self._apply_pagination(pairs, options.pagination)
        return pairs

    async def update(self, trading_pair: TradingPair) -> TradingPair:
        """Обновить торговую пару."""
        pair_id = str(trading_pair.symbol)
        if pair_id not in self._trading_pairs:
            raise EntityNotFoundError("TradingPair not found", "TradingPair", pair_id)
        self._trading_pairs[pair_id] = trading_pair
        # Инвалидация кэша
        await self.invalidate_cache(CacheKey(pair_id))
        return trading_pair

    async def delete(self, pair_id: Union[UUID, str]) -> bool:
        """Удалить торговую пару."""
        pair_id_str = str(pair_id)
        if pair_id_str not in self._trading_pairs:
            return False
        del self._trading_pairs[pair_id_str]
        # Инвалидация кэша
        await self.invalidate_cache(CacheKey(pair_id_str))
        return True

    async def exists(self, pair_id: Union[UUID, str]) -> bool:
        """Проверить существование торговой пары."""
        pair_id_str = str(pair_id)
        return pair_id_str in self._trading_pairs

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество торговых пар."""
        pairs = list(self._trading_pairs.values())
        if filters:
            pairs = self._apply_filters(pairs, filters)
        return len(pairs)

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[TradingPair]:
        """Потоковое получение торговых пар."""
        pairs = list(self._trading_pairs.values())
        if options and options.filters:
            pairs = self._apply_filters(pairs, options.filters)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            for pair in batch:
                yield pair

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Транзакция."""
        async def _transaction():
            # Для in-memory репозитория транзакции не нужны
            class MockTransaction(TransactionProtocol):
                async def __aenter__(self) -> "MockTransaction":
                    return self
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
                async def commit(self) -> None:
                    pass
                async def rollback(self) -> None:
                    pass
                async def is_active(self) -> bool:
                    return True
            yield MockTransaction()
        
        async with _transaction() as transaction:
            yield transaction

    async def execute_in_transaction(self, operation: Callable, *args, **kwargs) -> Any:
        """Выполнить операцию в транзакции."""
        return await operation(*args, **kwargs)

    async def bulk_save(self, trading_pairs: List[TradingPair]) -> BulkOperationResult:
        """Пакетное сохранение торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        for pair in trading_pairs:
            try:
                await self.save(pair)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append({"error": str(e), "entity_id": str(pair.symbol)})
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def bulk_update(
        self, trading_pairs: List[TradingPair]
    ) -> BulkOperationResult:
        """Пакетное обновление торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        for pair in trading_pairs:
            try:
                await self.update(pair)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append({"error": str(e), "entity_id": str(pair.symbol)})
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def bulk_delete(
        self, pair_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """Пакетное удаление торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        for pair_id in pair_ids:
            try:
                if await self.delete(pair_id):
                    success_count += 1
                else:
                    error_count += 1
                    errors.append({"error": f"Trading pair not found: {pair_id}", "entity_id": str(pair_id)})
            except Exception as e:
                error_count += 1
                errors.append({"error": str(e), "entity_id": str(pair_id)})
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def bulk_upsert(
        self, trading_pairs: List[TradingPair], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """Пакетное upsert торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        for pair in trading_pairs:
            try:
                if await self.exists(str(pair.symbol)):
                    await self.update(pair)
                else:
                    await self.save(pair)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append({"error": str(e), "entity_id": str(pair.symbol)})
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[TradingPair]:
        """Получить из кэша."""
        cache_key = CacheKey(str(key))
        if cache_key not in self._cache:
            return None
        if cache_key in self._cache_ttl and datetime.now() > self._cache_ttl[cache_key]:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
            return None
        return self._cache.get(cache_key)

    async def set_cache(
        self,
        key: Union[UUID, str],
        trading_pair: TradingPair,
        ttl: Optional[int] = None,
    ) -> None:
        """Установить в кэш."""
        cache_key = CacheKey(str(key))
        self._cache[cache_key] = trading_pair
        ttl_seconds = ttl or self._cache_ttl_seconds
        self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)
        await self._evict_cache()

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        cache_key = CacheKey(str(key))
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_ttl:
            del self._cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._cache_ttl.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        return RepositoryResponse(
            success=True,
            data={
                "total_entities": len(self._trading_pairs),
                "cache_size": len(self._cache),
                "cache_hit_rate": self._get_cache_hit_rate(),
                "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
            },
            total_count=len(self._trading_pairs),
        )

    @property
    def state(self) -> RepositoryStateProtocol:
        """Состояние репозитория."""
        return self._state

    async def find_by(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> List[TradingPair]:
        """Найти торговые пары по фильтрам."""
        pairs = list(self._trading_pairs.values())
        pairs = self._apply_filters(pairs, filters)
        if options and hasattr(options, 'sort_orders') and options.sort_orders:
            pairs = self._apply_sort(pairs, options.sort_orders)
        if options and options.pagination:
            pairs = self._apply_pagination(pairs, options.pagination)
        return pairs

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[TradingPair]:
        """Найти одну торговую пару по фильтрам."""
        pairs = await self.find_by(filters)
        return pairs[0] if pairs else None

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        try:
            return {
                "total_operations": len(self._trading_pairs),
                "successful_operations": len(self._trading_pairs),
                "failed_operations": 0,
                "average_response_time": 0.001,
                "cache_hit_rate": self._get_cache_hit_rate(),
                "uptime": (datetime.now() - self._startup_time).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "average_response_time": 0.0,
                "cache_hit_rate": 0.0,
                "uptime": 0.0,
            }

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "response_time": 1.0,
                "error_count": 0,
                "last_error": None,
                "uptime": (datetime.now() - self._startup_time).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Error in health check: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "response_time": 0.0,
                "error_count": 1,
                "last_error": str(e),
                "uptime": 0.0,
            }

    async def get_cache_stats(self) -> RepositoryResponse:
        """Получить статистику кэша."""
        return RepositoryResponse(
            success=True,
            data={
                "cache_size": len(self._cache),
                "cache_hit_rate": self._get_cache_hit_rate(),
                "cache_ttl_seconds": self._cache_ttl_seconds,
                "cache_max_size": self._cache_max_size,
            },
        )

    async def save_trading_pair(self, pair: TradingPair) -> bool:
        """Сохранить торговую пару."""
        try:
            await self.save(pair)
            return True
        except Exception:
            return False

    async def get_trading_pair(
        self, pair_id: Union[UUID, str]
    ) -> Optional[TradingPair]:
        """Получить торговую пару."""
        return await self.get_by_id(pair_id)

    async def get_all_trading_pairs(self) -> List[TradingPair]:
        """Получить все торговые пары."""
        return await self.get_all()

    def _apply_filters(
        self, pairs: List[TradingPair], filters: List[QueryFilter]
    ) -> List[TradingPair]:
        """Применить фильтры к торговым парам."""
        filtered_pairs = pairs
        for filter_obj in filters:
            if filter_obj.field == "symbol":
                filtered_pairs = [
                    p for p in filtered_pairs
                    if str(p.symbol) == str(filter_obj.value)
                ]
            elif filter_obj.field == "base_currency":
                filtered_pairs = [
                    p for p in filtered_pairs
                    if p.base_currency.symbol == str(filter_obj.value)
                ]
            elif filter_obj.field == "quote_currency":
                filtered_pairs = [
                    p for p in filtered_pairs
                    if p.quote_currency.symbol == str(filter_obj.value)
                ]
        return filtered_pairs

    def _apply_sort(
        self, pairs: List[TradingPair], sort_orders: List[Any]
    ) -> List[TradingPair]:
        """Применить сортировку к торговым парам."""
        for sort_order in sort_orders:
            if sort_order.field == "symbol":
                reverse = sort_order.direction == "desc"
                pairs.sort(key=lambda p: str(p.symbol), reverse=reverse)
            elif sort_order.field == "base_currency":
                reverse = sort_order.direction == "desc"
                pairs.sort(key=lambda p: p.base_currency.symbol, reverse=reverse)
        return pairs

    def _apply_pagination(
        self, pairs: List[TradingPair], pagination: Any
    ) -> List[TradingPair]:
        """Применить пагинацию к торговым парам."""
        if pagination.offset is not None and pagination.limit is not None:
            return pairs[pagination.offset:pagination.offset + pagination.limit]
        elif pagination.page and pagination.page_size:
            start = (pagination.page - 1) * pagination.page_size
            end = start + pagination.page_size
            return pairs[start:end]
        return pairs

    async def _evict_cache(self) -> None:
        """Вытеснить старые записи из кэша."""
        if not self._cache:
            return
        oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
        await self.invalidate_cache(oldest_key)

    async def _cleanup_cache(self) -> None:
        """Очистить истекшие записи кэша."""
        now = datetime.now()
        expired_keys = [
            key for key, ttl in self._cache_ttl.items() if now > ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(3600)  # Каждый час
                await self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {str(e)}")

    def _get_cache_hit_rate(self) -> float:
        """Получить hit rate кэша."""
        total_requests = len(self._cache) + len(self._trading_pairs)
        if total_requests == 0:
            return 0.0
        return len(self._cache) / total_requests


class PostgresTradingPairRepository(BaseRepository[TradingPair], TradingPairRepositoryProtocol):
    """
    PostgreSQL реализация репозитория торговых пар.
    """

    def __init__(self, connection_string: str):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection_string = connection_string
        self._pool: Optional[asyncpg.Pool] = None
        self._trading_pairs: Dict[str, TradingPair] = {}
        # Приводим типы кэша к совместимым с базовым классом
        self._cache: Dict[CacheKey, TradingPair] = {}
        self._cache_ttl: Dict[CacheKey, datetime] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        self._state = RepositoryStateProtocol.CONNECTED
        self._startup_time = datetime.now()
        asyncio.create_task(self._background_cleanup())
        self.logger.info("PostgresTradingPairRepository initialized")

    async def _get_pool(self) -> Any:
        """Получить пул соединений."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool

    async def save(self, trading_pair: TradingPair) -> TradingPair:
        """Сохранить торговую пару."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO trading_pairs (id, symbol, base_currency, quote_currency, status)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    base_currency = EXCLUDED.base_currency,
                    quote_currency = EXCLUDED.quote_currency,
                    status = EXCLUDED.status,
                    updated_at = NOW()
            """, str(trading_pair.symbol), str(trading_pair.symbol), 
                 trading_pair.base_currency.symbol, trading_pair.quote_currency.symbol, "TRADING")
        await self.invalidate_cache(CacheKey(str(trading_pair.symbol)))
        return trading_pair

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[TradingPair]:
        """Получить торговую пару по ID."""
        entity_id_str = str(entity_id)
        
        # Проверяем кэш
        cached = await self.get_from_cache(CacheKey(entity_id_str))
        if cached:
            return cached
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM trading_pairs WHERE id = $1",
                entity_id_str
            )
            if not row:
                raise EntityNotFoundError("TradingPair not found", "TradingPair", entity_id_str)
            
            trading_pair = self._row_to_trading_pair(row)
            await self.set_cache(CacheKey(entity_id_str), trading_pair)
            return trading_pair

    async def get_all(
        self, options: Optional[QueryOptions] = None
    ) -> List[TradingPair]:
        """Получить все торговые пары."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = "SELECT * FROM trading_pairs"
            params = []
            
            if options and options.filters:
                where_clauses = []
                for i, filter_obj in enumerate(options.filters, 1):
                    if filter_obj.field == "symbol":
                        where_clauses.append(f"symbol = ${i}")
                        params.append(str(filter_obj.value))
                    elif filter_obj.field == "status":
                        where_clauses.append(f"status = ${i}")
                        params.append(str(filter_obj.value))
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            if options and hasattr(options, 'sort_orders') and options.sort_orders:
                order_clauses = []
                for sort_order in options.sort_orders:
                    direction = "DESC" if sort_order.direction == "desc" else "ASC"
                    order_clauses.append(f"{sort_order.field} {direction}")
                query += " ORDER BY " + ", ".join(order_clauses)
            
            if options and options.pagination:
                if options.pagination.limit:
                    query += f" LIMIT {options.pagination.limit}"
                if options.pagination.offset:
                    query += f" OFFSET {options.pagination.offset}"
            
            rows = await conn.fetch(query, *params)
            return [self._row_to_trading_pair(row) for row in rows]

    async def update(self, trading_pair: TradingPair) -> TradingPair:
        """Обновить торговую пару."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE trading_pairs 
                SET symbol = $2, base_currency = $3, quote_currency = $4, updated_at = NOW()
                WHERE id = $1
            """, str(trading_pair.symbol), str(trading_pair.symbol),
                 trading_pair.base_currency.symbol, trading_pair.quote_currency.symbol)
            
            if result == "UPDATE 0":
                raise EntityNotFoundError("TradingPair not found", "TradingPair", str(trading_pair.symbol))
        
        await self.invalidate_cache(CacheKey(str(trading_pair.symbol)))
        return trading_pair

    async def delete(self, pair_id: Union[UUID, str]) -> bool:
        """Удалить торговую пару."""
        pair_id_str = str(pair_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM trading_pairs WHERE id = $1",
                pair_id_str
            )
            success = result == "DELETE 1"
            if success:
                await self.invalidate_cache(CacheKey(pair_id_str))
            return bool(success)

    async def exists(self, pair_id: Union[UUID, str]) -> bool:
        """Проверить существование торговой пары."""
        pair_id_str = str(pair_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM trading_pairs WHERE id = $1)",
                pair_id_str
            )
            return bool(result)

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчитать количество торговых пар."""
        try:
            pool = await self._get_pool()
            if not pool:
                return 0
            
            query = "SELECT COUNT(*) FROM trading_pairs"
            params = []
            
            if filters:
                where_conditions = []
                for filter_obj in filters:
                    if filter_obj.operator.value == "equals":
                        where_conditions.append(f"{filter_obj.field} = ${len(params) + 1}")
                        params.append(filter_obj.value)
                    elif filter_obj.operator.value == "like":
                        where_conditions.append(f"{filter_obj.field} LIKE ${len(params) + 1}")
                        params.append(f"%{filter_obj.value}%")
                    # Добавьте другие операторы по необходимости
                
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
            
            async with pool.acquire() as conn:
                result = await conn.fetchval(query, *params)
                return result or 0
        except Exception as e:
            self.logger.error(f"Error counting trading pairs: {str(e)}")
            return 0

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[TradingPair]:
        """Потоковое получение торговых пар."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = "SELECT * FROM trading_pairs"
            params = []
            
            if options and options.filters:
                where_clauses = []
                for i, filter_obj in enumerate(options.filters, 1):
                    if filter_obj.field == "symbol":
                        where_clauses.append(f"symbol = ${i}")
                        params.append(str(filter_obj.value))
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            query += f" LIMIT {batch_size}"
            
            async for row in conn.cursor(query, *params):
                yield self._row_to_trading_pair(row[0])

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Транзакция."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction() as transaction:
                class PostgresTransaction(TransactionProtocol):
                    def __init__(self, transaction):
                        self.transaction = transaction
                    
                    async def __aenter__(self) -> "PostgresTransaction":
                        return self
                    
                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass
                    
                    async def commit(self) -> None:
                        pass  # Транзакция коммитится автоматически
                    
                    async def rollback(self) -> None:
                        await self.transaction.rollback()
                    
                    async def is_active(self) -> bool:
                        return not self.transaction.is_closed()
                
                yield PostgresTransaction(transaction)

    async def execute_in_transaction(self, operation: Callable, *args, **kwargs) -> Any:
        """Выполнить операцию в транзакции."""
        async with self.transaction() as transaction:
            return await operation(*args, **kwargs)

    async def bulk_save(self, trading_pairs: List[TradingPair]) -> BulkOperationResult:
        """Пакетное сохранение торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for pair in trading_pairs:
                    try:
                        await conn.execute("""
                            INSERT INTO trading_pairs (id, symbol, base_currency, quote_currency, status)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (id) DO UPDATE SET
                                symbol = EXCLUDED.symbol,
                                base_currency = EXCLUDED.base_currency,
                                quote_currency = EXCLUDED.quote_currency,
                                status = EXCLUDED.status,
                                updated_at = NOW()
                        """, str(pair.symbol), str(pair.symbol),
                             pair.base_currency.symbol, pair.quote_currency.symbol, "TRADING")
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        errors.append({"error": str(e), "entity_id": str(pair.symbol)})
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def bulk_update(
        self, trading_pairs: List[TradingPair]
    ) -> BulkOperationResult:
        """Пакетное обновление торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for pair in trading_pairs:
                    try:
                        result = await conn.execute("""
                            UPDATE trading_pairs 
                            SET symbol = $2, base_currency = $3, quote_currency = $4, updated_at = NOW()
                            WHERE id = $1
                        """, str(pair.symbol), str(pair.symbol),
                             pair.base_currency.symbol, pair.quote_currency.symbol)
                        
                        if result == "UPDATE 0":
                            error_count += 1
                            errors.append({"error": f"Trading pair not found: {pair.symbol}", "entity_id": str(pair.symbol)})
                        else:
                            success_count += 1
                    except Exception as e:
                        error_count += 1
                        errors.append({"error": str(e), "entity_id": str(pair.symbol)})
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def bulk_delete(
        self, pair_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """Пакетное удаление торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for pair_id in pair_ids:
                    try:
                        result = await conn.execute(
                            "DELETE FROM trading_pairs WHERE id = $1",
                            str(pair_id)
                        )
                        if result == "DELETE 1":
                            success_count += 1
                        else:
                            error_count += 1
                            errors.append({"error": f"Trading pair not found: {pair_id}", "entity_id": str(pair_id)})
                    except Exception as e:
                        error_count += 1
                        errors.append({"error": str(e), "entity_id": str(pair_id)})
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def bulk_upsert(
        self, trading_pairs: List[TradingPair], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """Пакетное upsert торговых пар."""
        success_count = 0
        error_count = 0
        errors = []
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for pair in trading_pairs:
                    try:
                        await conn.execute("""
                            INSERT INTO trading_pairs (id, symbol, base_currency, quote_currency, status)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (id) DO UPDATE SET
                                symbol = EXCLUDED.symbol,
                                base_currency = EXCLUDED.base_currency,
                                quote_currency = EXCLUDED.quote_currency,
                                status = EXCLUDED.status,
                                updated_at = NOW()
                        """, str(pair.symbol), str(pair.symbol),
                             pair.base_currency.symbol, pair.quote_currency.symbol, "TRADING")
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        errors.append({"error": str(e), "entity_id": str(pair.symbol)})
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[TradingPair]:
        """Получить из кэша."""
        cache_key = CacheKey(str(key))
        if cache_key not in self._cache:
            return None
        if cache_key in self._cache_ttl and datetime.now() > self._cache_ttl[cache_key]:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
            return None
        return self._cache.get(cache_key)

    async def set_cache(
        self,
        key: Union[UUID, str],
        trading_pair: TradingPair,
        ttl: Optional[int] = None,
    ) -> None:
        """Установить в кэш."""
        cache_key = CacheKey(str(key))
        self._cache[cache_key] = trading_pair
        ttl_seconds = ttl or self._cache_ttl_seconds
        self._cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)
        await self._evict_cache()

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """Инвалидировать кэш."""
        cache_key = CacheKey(str(key))
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_ttl:
            del self._cache_ttl[cache_key]

    async def clear_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._cache_ttl.clear()

    async def get_repository_stats(self) -> RepositoryResponse:
        """Получить статистику репозитория."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            total_count = await conn.fetchval("SELECT COUNT(*) FROM trading_pairs")
            return RepositoryResponse(
                success=True,
                data={
                    "total_entities": total_count or 0,
                    "cache_size": len(self._cache),
                    "cache_hit_rate": self._get_cache_hit_rate(),
                    "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
                },
                total_count=total_count or 0,
            )

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """Получить метрики производительности."""
        try:
            return {
                "total_operations": len(self._trading_pairs),
                "successful_operations": len(self._trading_pairs),
                "failed_operations": 0,
                "average_response_time": 0.001,
                "cache_hit_rate": self._get_cache_hit_rate(),
                "uptime": (datetime.now() - self._startup_time).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "average_response_time": 0.0,
                "cache_hit_rate": 0.0,
                "uptime": 0.0,
            }

    async def health_check(self) -> HealthCheckDict:
        """Проверка здоровья репозитория."""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "response_time": 1.0,
                "error_count": 0,
                "last_error": None,
                "uptime": (datetime.now() - self._startup_time).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Error in health check: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "response_time": 0.0,
                "error_count": 1,
                "last_error": str(e),
                "uptime": 0.0,
            }

    async def save_trading_pair(self, pair: TradingPair) -> bool:
        """Сохранить торговую пару."""
        try:
            await self.save(pair)
            return True
        except Exception:
            return False

    async def get_trading_pair(
        self, pair_id: Union[UUID, str]
    ) -> Optional[TradingPair]:
        """Получить торговую пару."""
        return await self.get_by_id(pair_id)

    async def get_all_trading_pairs(self) -> List[TradingPair]:
        """Получить все торговые пары."""
        return await self.get_all()

    def _row_to_trading_pair(self, row: Any) -> TradingPair:
        """Преобразовать строку БД в TradingPair."""
        from domain.value_objects.currency import Currency
        from domain.types import Symbol
        
        base_currency = Currency.from_string(row.get("base_currency", "USD"))
        quote_currency = Currency.from_string(row.get("quote_currency", "USD"))
        
        return TradingPair(
            symbol=Symbol(row["symbol"]),
            base_currency=base_currency,
            quote_currency=quote_currency,
        )

    async def _evict_cache(self) -> None:
        """Эвакуация кэша."""
        if len(self._cache) > self._cache_max_size:
            # Удаляем старые записи
            sorted_items = sorted(
                self._cache_ttl.items(), key=lambda x: x[1]
            )
            items_to_remove = len(self._cache) - self._cache_max_size
            for key, _ in sorted_items[:items_to_remove]:
                del self._cache[key]
                del self._cache_ttl[key]

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        current_time = datetime.now()
        expired_keys = [
            key for key, ttl in self._cache_ttl.items()
            if current_time > ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            try:
                await asyncio.sleep(60)  # Каждую минуту
                await self._cleanup_cache()
                self._last_cleanup = datetime.now()
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")

    async def _initialize_pool(self) -> None:
        """Инициализация пула соединений."""
        try:
            self._pool = await asyncpg.create_pool(self.connection_string)
            self._state = RepositoryStateTypes.CONNECTED
            self.logger.info("Database pool initialized successfully")
        except Exception as e:
            self._state = RepositoryStateTypes.ERROR
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise

    def _get_cache_hit_rate(self) -> float:
        """Получить hit rate кэша."""
        total_requests = len(self._cache) + len(self._trading_pairs)
        if total_requests == 0:
            return 0.0
        return len(self._cache) / total_requests

    async def find_by(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> List[TradingPair]:
        """Найти торговые пары по фильтрам."""
        try:
            pool = await self._get_pool()
            if not pool:
                return []
            
            query = "SELECT * FROM trading_pairs"
            params = []
            
            if filters:
                where_conditions = []
                for filter_obj in filters:
                    if filter_obj.operator.value == "equals":
                        where_conditions.append(f"{filter_obj.field} = ${len(params) + 1}")
                        params.append(filter_obj.value)
                    elif filter_obj.operator.value == "like":
                        where_conditions.append(f"{filter_obj.field} LIKE ${len(params) + 1}")
                        params.append(f"%{filter_obj.value}%")
                    # Добавьте другие операторы по необходимости
                
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
            
            # Добавляем сортировку и пагинацию
            if options and hasattr(options, 'sort_orders') and options.sort_orders:
                sort_clauses = []
                for sort_order in options.sort_orders:
                    sort_clauses.append(f"{sort_order.field} {sort_order.direction}")
                if sort_clauses:
                    query += " ORDER BY " + ", ".join(sort_clauses)
            
            if options and options.pagination:
                offset = (options.pagination.page - 1) * options.pagination.page_size
                query += f" LIMIT {options.pagination.page_size} OFFSET {offset}"
            
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [self._row_to_trading_pair(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding trading pairs: {str(e)}")
            return []

    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[TradingPair]:
        """Найти одну торговую пару по фильтрам."""
        pairs = await self.find_by(filters)
        return pairs[0] if pairs else None
