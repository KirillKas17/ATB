"""
Реализация репозитория рыночных данных.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union, Coroutine, AsyncContextManager
from uuid import UUID

from domain.entities.market import MarketData, MarketRegime
from domain.exceptions.protocol_exceptions import EntityUpdateError
from domain.protocols.repository_protocol import (
    BulkOperationResult,
    MarketRepositoryProtocol,
    QueryOptions,
    RepositoryResponse,
    RepositoryState,
    TransactionProtocol,
    PerformanceMetricsDict,
    HealthCheckDict,
    QueryFilter,
    QueryOperator,
)
from domain.types import Symbol
from domain.protocols.repository_protocol import QueryFilter as ProtocolQueryFilter
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from contextlib import asynccontextmanager
from contextlib import AbstractAsyncContextManager, _AsyncGeneratorContextManager


class InMemoryMarketRepository(MarketRepositoryProtocol):
    """
    Сверхпродвинутая in-memory реализация репозитория рыночных данных.
    - Кэширование с TTL
    - Асинхронные транзакции
    - Индексация по символу, timeframe, времени
    - Аналитика рыночных данных
    - Мониторинг и health-check
    """

    def __init__(self) -> None:
        self._market_data: Dict[str, List[MarketData]] = defaultdict(list)
        self._market_regimes: Dict[str, MarketRegime] = {}
        self._data_by_timeframe: Dict[str, Dict[str, List[MarketData]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._latest_data: Dict[str, MarketData] = {}
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._cache_max_size: int = 1000
        self._cache_ttl_seconds: int = 300
        self._metrics: Dict[str, Any] = {
            "total_market_data": 0,
            "total_symbols": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup": datetime.now(),
        }
        self._state: RepositoryState = RepositoryState.CONNECTED
        self._startup_time: datetime = datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def save_market_data(self, market_data: MarketData) -> MarketData:
        symbol: str = str(market_data.symbol)
        self._market_data[symbol].append(market_data)
        self._data_by_timeframe[str(market_data.timeframe)][symbol].append(market_data)
        self._latest_data[symbol] = market_data
        self._metrics["total_market_data"] = sum(
            len(lst) for lst in self._market_data.values()
        )
        self._metrics["total_symbols"] = len(self._market_data)
        await self.invalidate_cache(f"market_data:{symbol}")
        return market_data

    async def get_market_data(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MarketData]:
        symbol_str: str = str(symbol)
        cache_key: str = f"market_data:{symbol_str}:{timeframe}:{limit}"
        cached: Optional[List[MarketData]] = await self.get_from_cache_list(cache_key)
        if cached:
            self._metrics["cache_hits"] = int(self._metrics.get("cache_hits", 0)) + 1
            return cached
        self._metrics["cache_misses"] = int(self._metrics.get("cache_misses", 0)) + 1
        data: List[MarketData] = self._data_by_timeframe[timeframe].get(symbol_str, [])
        if start_date:
            data = [d for d in data if d.timestamp >= start_date]
        if end_date:
            data = [d for d in data if d.timestamp <= end_date]
        result: List[MarketData] = data[-limit:] if limit > 0 else data
        await self.set_cache_list(cache_key, result, ttl=60)
        return result

    async def get_latest_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        """Получение последних рыночных данных."""
        return self._latest_data.get(str(symbol))

    async def get_market_data_by_time_range(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[MarketData]:
        """Получение рыночных данных по временному диапазону."""
        try:
            data = self._market_data.get(symbol, [])
            filtered_data = [
                item for item in data if start_time <= item.timestamp <= end_time
            ]
            return filtered_data
        except Exception as e:
            self.logger.error(
                f"Failed to get market data by time range for {symbol}: {e}"
            )
            return []

    async def save(self, entity: MarketData) -> MarketData:
        """Сохранить сущность."""
        if isinstance(entity, MarketData):
            return await self.save_market_data(entity)
        raise EntityUpdateError(f"Unknown entity type: {type(entity)}")

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[MarketData]:
        """Получить сущность по ID."""
        # Для рыночных данных обычно используется символ, а не UUID
        return None

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[MarketData]:
        """Получить все сущности."""
        all_data: List[MarketData] = []
        for symbol_data in self._market_data.values():
            all_data.extend(symbol_data)
        return all_data

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить сущность."""
        # Для рыночных данных обычно используется символ, а не UUID
        entity_id_str = str(entity_id)
        if entity_id_str in self._market_data:
            del self._market_data[entity_id_str]
            return True
        return False

    # Дополнительные методы для совместимости
    async def get_market_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Получить рыночный режим для символа."""
        return self._market_regimes.get(symbol)

    async def bulk_save_market_data(
        self, market_data_list: List[MarketData]
    ) -> BulkOperationResult:
        """Массовое сохранение рыночных данных."""
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        
        for market_data in market_data_list:
            try:
                await self.save_market_data(market_data)
                processed_ids.append(market_data.id)
            except Exception as e:
                errors.append({"entity_id": str(market_data.id), "error": str(e)})
        
        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=len(processed_ids),
            error_count=len(errors)
        )

    async def cleanup_old_data(self, symbol: Symbol, older_than: datetime) -> int:
        symbol_str = str(symbol)
        if symbol_str not in self._market_data:
            return 0

        original_count = len(self._market_data[symbol_str])
        self._market_data[symbol_str] = [
            d for d in self._market_data[symbol_str] if d.timestamp >= older_than
        ]
        removed_count = original_count - len(self._market_data[symbol_str])

        # Обновляем индексы
        for timeframe in self._data_by_timeframe:
            if symbol_str in self._data_by_timeframe[timeframe]:
                self._data_by_timeframe[timeframe][symbol_str] = [
                    d
                    for d in self._data_by_timeframe[timeframe][symbol_str]
                    if d.timestamp >= older_than
                ]

        return removed_count

    async def save_market_regime(self, symbol: str, regime: MarketRegime) -> bool:
        """Сохранение рыночного режима."""
        try:
            self._market_regimes[symbol] = regime
            await self.invalidate_cache(f"market_regime:{symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save market regime for {symbol}: {e}")
            return False

    async def update(self, entity: MarketData) -> MarketData:
        if isinstance(entity, MarketData):
            return await self.save_market_data(entity)
        raise EntityUpdateError(f"Unknown entity type: {type(entity)}")

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        return await self.delete(entity_id)

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        return True

    async def find_by(
        self, filters: List[ProtocolQueryFilter], options: Optional[QueryOptions] = None
    ) -> List[MarketData]:
        all_data = await self.get_all(options)
        for f in filters:
            if f.field == "symbol":
                all_data = [d for d in all_data if str(d.symbol) == f.value]
            if f.field == "timeframe":
                all_data = [d for d in all_data if d.timeframe == f.value]
        return all_data

    async def find_one_by(self, filters: List[ProtocolQueryFilter]) -> Optional[MarketData]:
        results = await self.find_by(filters)
        return results[0] if results else None

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        return False

    async def count(self, filters: Optional[List[ProtocolQueryFilter]] = None) -> int:
        if filters:
            data = await self.find_by(filters)
            return len(data)
        return sum(len(data) for data in self._market_data.values())

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[MarketData]:
        """Потоковая обработка рыночных данных."""
        data_list: List[MarketData] = []
        for symbol_data in self._market_data.values():
            data_list.extend(symbol_data)
        
        if options and options.filters:
            data_list = self._apply_filters(data_list, options.filters)
        if options and options.sort_orders:
            data_list = self._apply_sort(data_list, options.sort_orders)
        if options and options.pagination:
            data_list = self._apply_pagination(data_list, options.pagination)
        
        for data in data_list:
            yield data

    def _apply_filters(self, data_list: List[MarketData], filters: List[ProtocolQueryFilter]) -> List[MarketData]:
        """Применение фильтров к данным."""
        filtered_data = data_list
        for f in filters:
            if f.field == "symbol":
                filtered_data = [d for d in filtered_data if str(d.symbol) == f.value]
            elif f.field == "timeframe":
                filtered_data = [d for d in filtered_data if d.timeframe == f.value]
        return filtered_data

    def _apply_sort(self, data_list: List[MarketData], sort_orders: List[Any]) -> List[MarketData]:
        """Применение сортировки к данным."""
        sorted_data = data_list.copy()
        for sort_order in reversed(sort_orders):
            reverse = getattr(sort_order, 'direction', 'asc') == 'desc'
            field = getattr(sort_order, 'field', 'timestamp')
            sorted_data.sort(
                key=lambda x: getattr(x, field, x.timestamp),
                reverse=reverse
            )
        return sorted_data

    def _apply_pagination(self, data_list: List[MarketData], pagination: Any) -> List[MarketData]:
        """Применение пагинации к данным."""
        page = getattr(pagination, 'page', 1)
        page_size = getattr(pagination, 'page_size', 100)
        start = (page - 1) * page_size
        end = start + page_size
        return data_list[start:end]

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Транзакция."""
        class MarketTransaction(TransactionProtocol):
            def __init__(self, repo: "InMemoryMarketRepository") -> None:
                self.repo = repo
                self.active = True
                
            async def __aenter__(self) -> "MarketTransaction":
                return self
                
            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if exc_type is not None:
                    await self.rollback()
                    
            async def commit(self) -> None:
                pass
                
            async def rollback(self) -> None:
                self.active = False
                
            async def is_active(self) -> bool:
                return self.active
        
        yield MarketTransaction(self)

    async def execute_in_transaction(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнение операции в транзакции."""
        async with self.transaction() as txn:
            return await operation(*args, **kwargs)

    async def bulk_save(self, entities: List[MarketData]) -> BulkOperationResult:
        processed_ids: List[Union[UUID, str]] = []
        errors: List[Dict[str, Any]] = []
        for entity in entities:
            try:
                await self.save(entity)
                processed_ids.append(entity.id)
            except Exception as e:
                errors.append({"entity_id": str(entity.id), "error": str(e)})
        return BulkOperationResult(
            processed_ids=processed_ids,
            errors=errors,
            success_count=len(processed_ids),
            error_count=len(errors)
        )

    async def bulk_update(self, entities: List[MarketData]) -> BulkOperationResult:
        return await self.bulk_save(entities)

    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        success, error, errors, ids = 0, 0, [], []
        for entity_id in entity_ids:
            try:
                if await self.delete(entity_id):
                    success += 1
                    ids.append(entity_id)
                else:
                    error += 1
                    errors.append(
                        {"entity_id": str(entity_id), "error": "Entity not found"}
                    )
            except Exception as ex:
                error += 1
                errors.append({"entity_id": str(entity_id), "error": str(ex)})
        return BulkOperationResult(
            success_count=success, error_count=error, errors=errors, processed_ids=ids
        )

    async def bulk_upsert(
        self, entities: List[MarketData], conflict_fields: List[str]
    ) -> BulkOperationResult:
        return await self.bulk_save(entities)

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[MarketData]:
        if key in self._cache:
            if key in self._cache_ttl and self._cache_ttl[key] < datetime.now():
                del self._cache[key]
                del self._cache_ttl[key]
                return None
            cached_value = self._cache[key]
            if isinstance(cached_value, MarketData):
                return cached_value
        return None

    async def get_from_cache_list(self, key: Union[UUID, str]) -> Optional[List[MarketData]]:
        if key in self._cache:
            if key in self._cache_ttl and self._cache_ttl[key] < datetime.now():
                del self._cache[key]
                del self._cache_ttl[key]
                return None
            cached_value = self._cache[key]
            if isinstance(cached_value, list):
                return cached_value
        return None

    async def set_cache(
        self, key: Union[UUID, str], entity: MarketData, ttl: Optional[int] = None
    ) -> None:
        if len(self._cache) >= self._cache_max_size:
            await self._evict_cache()
        self._cache[key] = entity
        ttl_value = ttl if ttl is not None else getattr(self, 'cache_ttl_seconds', 300)
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl_value)

    async def set_cache_list(
        self, key: Union[UUID, str], entities: List[MarketData], ttl: Optional[int] = None
    ) -> None:
        if len(self._cache) >= self._cache_max_size:
            await self._evict_cache()
        # Сохраняем список MarketData
        self._cache[key] = entities
        ttl_value = ttl if ttl is not None else getattr(self, 'cache_ttl_seconds', 300)
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl_value)

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_ttl:
            del self._cache_ttl[key]

    async def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_ttl.clear()

    async def _evict_cache(self) -> None:
        """Удалить устаревшие записи из кэша."""
        current_time = datetime.now()
        expired_keys = [
            key for key, ttl in self._cache_ttl.items() 
            if current_time > ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        if len(self._cache) > self._cache_max_size:
            # Удаляем самые старые записи
            sorted_keys = sorted(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
            keys_to_remove = sorted_keys[:len(self._cache) - self._cache_max_size]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_ttl[key]
            self.logger.info(f"Cleaned up {len(keys_to_remove)} cache entries due to size limit")

    async def _background_cleanup(self) -> None:
        """Фоновая очистка кэша."""
        while True:
            await asyncio.sleep(300)  # 5 минут
            await self._cleanup_cache()


class PostgresMarketRepository(MarketRepositoryProtocol):
    """
    PostgreSQL реализация репозитория рыночных данных.
    """

    def __init__(self, connection_string: str, cache_service: Optional[Any] = None) -> None:
        self.connection_string = connection_string
        self.cache_service = cache_service
        self._pool: Optional[Any] = None
        self._state = RepositoryState.DISCONNECTED
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _get_pool(self) -> Optional[Any]:
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(
                    self.connection_string, min_size=5, max_size=20
                )
                self._state = RepositoryState.CONNECTED
            except Exception as e:
                self._state = RepositoryState.ERROR
                self.logger.error(f"Failed to create connection pool: {e}")
                raise
        return self._pool

    async def _execute_with_retry(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pool = await self._get_pool()
                if pool is None:
                    raise Exception("Failed to get connection pool")
                async with pool.acquire() as conn:
                    result = await operation(conn, *args, **kwargs)
                    return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2**attempt))

    async def save_market_data(self, market_data: MarketData) -> MarketData:
        async def _save_operation(conn: Any) -> MarketData:
            query = """
                INSERT INTO market_data (id, symbol, timeframe, timestamp, open, high, low, close, volume, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    timeframe = EXCLUDED.timeframe,
                    timestamp = EXCLUDED.timestamp,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    metadata = EXCLUDED.metadata
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                str(market_data.id),
                str(market_data.symbol),
                str(market_data.timeframe),
                market_data.timestamp,
                float(market_data.open.value),
                float(market_data.high.value),
                float(market_data.low.value),
                float(market_data.close.value),
                float(market_data.volume.value),
                market_data.metadata,
            )
            return self._row_to_market_data(row)

        result = await self._execute_with_retry(_save_operation)
        if isinstance(result, MarketData):
            return result
        raise ValueError("Expected MarketData result")

    async def get_market_data(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MarketData]:
        async def _get_operation(conn: Any) -> List[MarketData]:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = $1 AND timeframe = $2
            """
            params = [str(symbol), timeframe]
            param_count = 2

            if start_date:
                param_count += 1
                query += f" AND timestamp >= ${param_count}"
                params.append(str(start_date))

            if end_date:
                param_count += 1
                query += f" AND timestamp <= ${param_count}"
                params.append(str(end_date))

            query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
            params.append(str(limit))

            rows = await conn.fetch(query, *params)
            return [self._row_to_market_data(row) for row in rows]

        result = await self._execute_with_retry(_get_operation)
        if isinstance(result, list):
            return result
        raise ValueError("Expected list result")

    async def get_latest_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        async def _get_operation(conn: Any) -> Optional[MarketData]:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = $1 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            row = await conn.fetchrow(query, str(symbol))
            return self._row_to_market_data(row) if row else None

        result = await self._execute_with_retry(_get_operation)
        if isinstance(result, MarketData) or result is None:
            return result
        raise ValueError("Expected MarketData or None result")

    async def save(self, entity: MarketData) -> MarketData:
        if isinstance(entity, MarketData):
            return await self.save_market_data(entity)
        raise EntityUpdateError(f"Unknown entity type: {type(entity)}")

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[MarketData]:
        async def _get_operation(conn: Any) -> Optional[MarketData]:
            query = "SELECT * FROM market_data WHERE id = $1"
            row = await conn.fetchrow(query, str(entity_id))
            return self._row_to_market_data(row) if row else None

        result = await self._execute_with_retry(_get_operation)
        if isinstance(result, MarketData) or result is None:
            return result
        raise ValueError("Expected MarketData or None result")

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[MarketData]:
        async def _get_operation(conn: Any) -> List[MarketData]:
            query = "SELECT * FROM market_data ORDER BY timestamp DESC"
            if options and hasattr(options, 'pagination') and options.pagination and options.pagination.limit:
                query += f" LIMIT {options.pagination.limit}"
            rows = await conn.fetch(query)
            return [self._row_to_market_data(row) for row in rows]

        result = await self._execute_with_retry(_get_operation)
        if isinstance(result, list):
            return result
        raise ValueError("Expected list result")

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        async def _delete_operation(conn: Any) -> bool:
            result = await conn.execute(
                "DELETE FROM market_data WHERE id = $1",
                entity_id
            )
            return bool(result == "DELETE 1")
        pool = await self._get_pool()
        if pool is None:
            return False
        result = await self._execute_with_retry(_delete_operation)
        return bool(result)

    async def get_market_regime(self, symbol: str) -> Optional[MarketRegime]:
        async def _get_operation(conn: Any) -> Optional[MarketRegime]:
            query = """
                SELECT * FROM market_regimes 
                WHERE symbol = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            row = await conn.fetchrow(query, symbol)
            return self._row_to_market_regime(row) if row else None

        result = await self._execute_with_retry(_get_operation)
        if isinstance(result, MarketRegime) or result is None:
            return result
        raise ValueError("Expected MarketRegime or None result")

    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        async def _delete_operation(conn: Any) -> BulkOperationResult:
            success_count = 0
            error_count = 0
            errors = []
            processed_ids: List[Union[UUID, str]] = []
            
            for entity_id in entity_ids:
                try:
                    result = await conn.execute(
                        "DELETE FROM market_data WHERE id = $1",
                        entity_id
                    )
                    if result == "DELETE 1":
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
        
        pool = await self._get_pool()
        if pool is None:
            return BulkOperationResult(
                processed_ids=[],
                errors=[],
                success_count=0,
                error_count=len(entity_ids)
            )
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_delete_operation)
        )
        return result if isinstance(result, BulkOperationResult) else BulkOperationResult(
            processed_ids=[],
            errors=[],
            success_count=0,
            error_count=len(entity_ids)
        )

    async def bulk_save(self, entities: List[MarketData]) -> BulkOperationResult:
        async def _save_operation(conn: Any) -> BulkOperationResult:
            success_count = 0
            error_count = 0
            errors = []
            processed_ids: List[Union[UUID, str]] = []
            
            for entity in entities:
                try:
                    await conn.execute(
                        """
                        INSERT INTO market_data (id, symbol, timeframe, open, high, low, close, volume, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        entity.id,
                        str(entity.symbol),
                        entity.timeframe,
                        float(entity.open.to_decimal()),
                        float(entity.high.to_decimal()),
                        float(entity.low.to_decimal()),
                        float(entity.close.to_decimal()),
                        float(entity.volume.to_decimal()),
                        entity.timestamp
                    )
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
        
        pool = await self._get_pool()
        if pool is None:
            return BulkOperationResult(
                processed_ids=[],
                errors=[],
                success_count=0,
                error_count=len(entities)
            )
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_save_operation)
        )
        return result if isinstance(result, BulkOperationResult) else BulkOperationResult(
            processed_ids=[],
            errors=[],
            success_count=0,
            error_count=len(entities)
        )

    async def bulk_save_market_data(
        self, market_data_list: List[MarketData]
    ) -> BulkOperationResult:
        return await self.bulk_save(market_data_list)

    async def bulk_update(self, entities: List[MarketData]) -> BulkOperationResult:
        return await self.bulk_save(entities)

    async def bulk_upsert(
        self, entities: List[MarketData], conflict_fields: List[str]
    ) -> BulkOperationResult:
        return await self.bulk_save(entities)

    async def cleanup_old_data(self, symbol: Symbol, older_than: datetime) -> int:
        async def _cleanup_operation(conn: Any) -> int:
            query = "DELETE FROM market_data WHERE symbol = $1 AND timestamp < $2"
            result = await conn.execute(query, str(symbol), older_than)
            return int(result.split()[1]) if result.startswith("DELETE") else 0

        result = await self._execute_with_retry(_cleanup_operation)
        if isinstance(result, int):
            return result
        raise ValueError("Expected int result")

    async def clear_cache(self) -> None:
        if self.cache_service:
            await self.cache_service.clear()

    async def count(self, filters: Optional[List[ProtocolQueryFilter]] = None) -> int:
        async def _count_operation(conn: Any) -> int:
            if filters:
                where_clause, params = self._build_where_clause(filters)
                query = f"SELECT COUNT(*) FROM market_data WHERE {where_clause}"
                result = await conn.fetchval(query, *params)
            else:
                result = await conn.fetchval("SELECT COUNT(*) FROM market_data")
            return result or 0
        
        pool = await self._get_pool()
        if pool is None:
            return 0
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_count_operation)
        )
        return int(result) if isinstance(result, int) else 0

    async def execute_in_transaction(
        self, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        async def _transaction_operation(conn: Any) -> Any:
            async with conn.transaction():
                return await operation(*args, **kwargs)

        pool = await self._get_pool()
        if pool is None:
            raise Exception("Failed to get connection pool")
        async with pool.acquire() as conn:
            return await _transaction_operation(conn)

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        async def _exists_operation(conn: Any) -> bool:
            query = "SELECT EXISTS(SELECT 1 FROM market_data WHERE id = $1)"
            result = await conn.fetchval(query, str(entity_id))
            return result or False

        result = await self._execute_with_retry(_exists_operation)
        if isinstance(result, bool):
            return result
        raise ValueError("Expected bool result")

    async def find_by(
        self, filters: List[ProtocolQueryFilter], options: Optional[QueryOptions] = None
    ) -> List[MarketData]:
        async def _find_operation(conn: Any) -> List[MarketData]:
            where_clause, params = self._build_where_clause(filters)
            query = f"SELECT * FROM market_data WHERE {where_clause}"
            
            if options and options.sort_orders:
                query += self._build_order_clause(options.sort_orders)
            
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            
            rows = await conn.fetch(query, *params)
            return [self._row_to_market_data(row) for row in rows]
        
        pool = await self._get_pool()
        if pool is None:
            return []
        result = await self._execute_with_retry(
            lambda: pool.acquire().__aenter__().then(_find_operation)
        )
        return result if isinstance(result, list) else []

    async def find_one_by(self, filters: List[ProtocolQueryFilter]) -> Optional[MarketData]:
        models = await self.find_by(filters)
        return models[0] if models else None

    async def get_cache_stats(self) -> RepositoryResponse:
        if self.cache_service:
            stats = await self.cache_service.get_stats()
            return RepositoryResponse(
                success=True,
                data=stats,
            )
        return RepositoryResponse(
            success=True, data={}
        )

    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[MarketData]:
        if self.cache_service:
            result = await self.cache_service.get(str(key))
            return result if isinstance(result, MarketData) else None
        return None

    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        pool = await self._get_pool()
        if pool is None:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.0",
                "profit_factor": "0.0",
                "sharpe_ratio": "0.0",
                "max_drawdown": "0.0",
                "total_return": "0.0",
                "average_trade": "0.0"
            }
        
        async with pool.acquire() as conn:
            total_entities = await conn.fetchval("SELECT COUNT(*) FROM market_data")
            return {
                "total_trades": total_entities or 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.0",
                "profit_factor": "0.0",
                "sharpe_ratio": "0.0",
                "max_drawdown": "0.0",
                "total_return": "0.0",
                "average_trade": "0.0"
            }

    async def get_repository_stats(self) -> RepositoryResponse:
        pool = await self._get_pool()
        if pool is None:
            return RepositoryResponse(success=False, data={})
        
        async with pool.acquire() as conn:
            total_entities = await conn.fetchval("SELECT COUNT(*) FROM market_data")
            stats = {
                "total_entities": total_entities or 0,
                "cache_hit_rate": 0.0,
                "avg_query_time": 0.0,
                "error_rate": 0.0
            }
            return RepositoryResponse(success=True, data=stats)

    async def health_check(self) -> HealthCheckDict:
        try:
            pool = await self._get_pool()
            if pool is None:
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "response_time": 0.0,
                    "error_count": 1,
                    "last_error": "Failed to get connection pool",
                    "uptime": 0.0
                }
            
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "response_time": 0.0,
                    "error_count": 0,
                    "last_error": None,
                    "uptime": 0.0
                }
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "response_time": 0.0,
                "error_count": 1,
                "last_error": str(e),
                "uptime": 0.0
            }

    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        if self.cache_service:
            await self.cache_service.delete(str(key))

    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        return True

    async def set_cache(
        self, key: Union[UUID, str], entity: MarketData, ttl: Optional[int] = None
    ) -> None:
        if self.cache_service:
            await self.cache_service.set(str(key), entity, ttl)

    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        return await self.delete(entity_id)

    async def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[MarketData]:
        """Потоковое чтение рыночных данных."""
        pool = await self._get_pool()
        if pool is None:
            return
        
        async with pool.acquire() as conn:
            query = "SELECT * FROM market_data"
            if options and options.pagination:
                query += f" LIMIT {options.pagination.page_size} OFFSET {(options.pagination.page - 1) * options.pagination.page_size}"
            
            rows = await conn.fetch(query)
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                for row in batch:
                    yield self._row_to_market_data(row)
                await asyncio.sleep(0)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TransactionProtocol]:
        """Контекстный менеджер для транзакций."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                class PostgresMarketTransaction(TransactionProtocol):
                    def __init__(self, conn: Any) -> None:
                        self.conn = conn
                        self._active = True
                    async def __aenter__(self) -> "PostgresMarketTransaction":
                        return self
                    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                        self._active = False
                    async def commit(self) -> None:
                        pass
                    async def rollback(self) -> None:
                        self._active = False
                    async def is_active(self) -> bool:
                        return self._active
                yield PostgresMarketTransaction(connection)

    async def update(self, entity: MarketData) -> MarketData:
        return await self.save_market_data(entity)

    def _row_to_market_data(self, row: Any) -> MarketData:
        """Преобразовать строку БД в объект MarketData."""
        from domain.value_objects.currency import Currency
        
        return MarketData(
            id=row["id"],
            symbol=Symbol(row["symbol"]),
            timeframe=row["timeframe"],
            timestamp=row["timestamp"],
            open=Price(Decimal(str(row["open"])), Currency.USD),
            high=Price(Decimal(str(row["high"])), Currency.USD),
            low=Price(Decimal(str(row["low"])), Currency.USD),
            close=Price(Decimal(str(row["close"])), Currency.USD),
            volume=Volume(Decimal(str(row["volume"])), Currency.USD),
            metadata=row.get("metadata", {})
        )

    def _row_to_market_regime(self, row: Any) -> Optional[MarketRegime]:
        """Преобразовать строку БД в объект MarketRegime."""
        if not row:
            return None
        try:
            return MarketRegime(row["regime_type"])
        except (KeyError, ValueError):
            return MarketRegime.UNKNOWN

    def _build_where_clause(self, filters: List[ProtocolQueryFilter]) -> tuple[str, List[Any]]:
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
            field = getattr(sort_order, 'field', 'timestamp') # Use getattr to handle different sort order types
            order_parts.append(f"{field} {direction}")
        
        return f" ORDER BY {', '.join(order_parts)}"

    async def close(self) -> None:
        """Закрыть соединения."""
        if self._pool:
            await self._pool.close()
